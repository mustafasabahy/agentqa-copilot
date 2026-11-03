from __future__ import annotations
import asyncio
import hashlib
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import Field
from .config import DB
from .models import Agent, AgentProfile, Contract, Evidence, Mode, Plan, Risk, Suite, TestCase, now, uid
from .store import Store
from .security import PolicyError, URLPolicy, redact
from .adapters import adapter_for
from .discovery import fingerprint
from .generation import coverage, generate, prioritize, validate_case, deduplicate, quality_review
from .capabilities import execution_capabilities
from .policy import Policy
from .runner import Runner
from .bootstrap import seed
from .analysis import calibration, clusters, compare, html_report, markdown_report, report
from . import llm, langsmith_integration as smith, provider, remote_approval, draft_workflow, offline_demo, judge_contract, compiler_demo

log=logging.getLogger('agentqa')

class OfflineDraftDemo(Contract):
    agent_id:str
    count:draft_workflow.CaseCount=3
    scenario:Literal["success","malformed","partial","duplicate","unsupported"]="success"
class ProviderPlanRequest(Contract):
    kind:Literal["initial","minimal","minimal_generation","minimal_judge"]="initial"
    agent_id:str
    trace:bool=False
    retry_of:str|None=None
class ProviderApprovalRequest(Contract):
    digest:str
    confirmation:Literal["I approve this exact provider plan"]
class RemotePlanRequest(Contract):
    action:Literal["projects","trace_read","import","selected","dataset","feedback","experiment","tracing"]
    payload:dict
class RemoteApprovalRequest(Contract):
    digest:str
    confirmation:Literal["I approve this exact remote plan"]
class SelectedImport(Contract):
    run_ids:list[str]=Field(min_length=1,max_length=3)
    remote_plan_id:str
class GenerateRequest(Contract):
    provider_plan_id:str|None=None
    count:int=Field(default=30,ge=1,le=1000)
    focus:str=''
    ai:bool=False
    budget:float=Field(default=0,ge=0,allow_inf_nan=False)
class PlanRequest(Contract):
    agent_id:str
    suite_id:str
    test_ids:list[str]
    mode:Mode
    budget:float=Field(default=0,ge=0,allow_inf_nan=False)
    concurrency:int=Field(default=1,ge=1,le=8)
    timeout:float=Field(default=30,ge=0.1,le=300)
    judge:bool=False
    evaluation_budget:float=Field(default=0,ge=0,allow_inf_nan=False)
    rate_per_second:float=Field(default=2,gt=0,le=20)
class ApprovalRequest(Contract):
    digest:str
    additional_authorization:bool=False
    confirmation:Literal['I approve this exact plan']
class ChatRequest(Contract):
    provider_plan_id:str|None=None
    agent_id:str
    session_id:str|None=None
    message:str=Field(min_length=1,max_length=8000)
    budget:float=Field(default=1,ge=0,le=100,allow_inf_nan=False)
class TraceImport(Contract):
    remote_plan_id:str|None=None
    project:str
    limit:int=Field(default=20,ge=1,le=100)
class GoldenJudgeRequest(Contract):
    provider_plan_id:str|None=None
    budget:float=Field(ge=0,le=100,allow_inf_nan=False)
class GoldenUpdate(Contract):
    reference:Literal['PASS','FAIL','INCONCLUSIVE','ERROR','SKIPPED']
    judge_verdict:Literal['PASS','FAIL','INCONCLUSIVE','ERROR','SKIPPED']|None=None
    notes:str
    evidence:dict=Field(default_factory=dict)
class OverrideRequest(Contract):
    result_id:str
    verdict:Literal['PASS','FAIL','INCONCLUSIVE','ERROR','SKIPPED']
    reason:str=Field(min_length=5)
class DatasetRequest(Contract):
    remote_plan_id:str|None=None
    name:str=Field(min_length=1)
    suite_id:str
class FeedbackRequest(Contract):
    remote_plan_id:str|None=None
    trace_id:str
    key:str
    score:float=Field(ge=0,le=1)
    comment:str
class ExperimentRequest(Contract):
    remote_plan_id:str|None=None
    name:str
    run_id:str


def create_app(db_path=None,admin_token=None,seed_data=True):
    store=Store(db_path or DB); runner=Runner(store); policy=Policy(store)
    expected=admin_token if admin_token is not None else os.getenv('AGENTQA_ADMIN_TOKEN','')
    @asynccontextmanager
    async def lifespan(app):
        # Interrupted runs are not silently resumed; old approvals are not replayed.
        for r in store.all('run'):
            if r['status'] in ('running','queued'):
                r['status']='interrupted'; r['ended_at']=now(); store.put('run',r)
        if seed_data: await seed(store)
        yield
        tasks=list(runner.tasks.values())
        for task in tasks: task.cancel()
        if tasks: await asyncio.gather(*tasks,return_exceptions=True)
    app=FastAPI(title='AgentQA Copilot',version='0.1.0',lifespan=lifespan)
    app.state.store=store; app.state.runner=runner
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
    @app.middleware('http')
    async def browser_boundary(request,call_next):
        origin=request.headers.get('origin')
        if origin and origin!=str(request.base_url).rstrip('/'):
            return JSONResponse({'detail':'Cross-origin access denied'},status_code=403)
        if int(request.headers.get('content-length','0'))>4_000_000:
            return JSONResponse({'detail':'Request too large'},status_code=413)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response
    async def user(authorization:str=Header(default='')):
        if not expected: raise HTTPException(503,'Configure AGENTQA_ADMIN_TOKEN using scripts/create_admin.py')
        if not authorization.startswith('Bearer ') or not hmac.compare_digest(authorization[7:],expected): raise HTTPException(401,'Authentication required')
        return 'local-admin'
    @app.exception_handler(PolicyError)
    async def policy_error(request,exc): return JSONResponse({'detail':str(exc)},status_code=403)
    @app.exception_handler(draft_workflow.DraftIncomplete)
    async def incomplete_drafts(request,exc):return JSONResponse({'detail':str(exc),'draft_request':exc.summary},status_code=422)
    @app.exception_handler(ValueError)
    async def bad_input(request,exc): return JSONResponse({'detail':redact(str(exc))[:1000]},status_code=422)
    @app.exception_handler(KeyError)
    async def missing(request,exc): return JSONResponse({'detail':'Record not found'},status_code=404)
    @app.exception_handler(Exception)
    async def unexpected(request,exc):
        log.error('Request failed: %s',type(exc).__name__)
        return JSONResponse({'detail':'Operation failed. Provider/network configuration may be unavailable; no secrets returned.'},status_code=502)
    def agent(id): return Agent.model_validate(store.get('agent',id))
    def suite(id): return Suite.model_validate(store.get('suite',id))
    @app.get('/health')
    def health(): return {'status':'ok','service':'AgentQA Copilot'}
    @app.get('/api/status')
    def status(actor=Depends(user)):
        return {'provider':provider.status(store),'actor':actor,'model':llm.MODEL,'openai_configured':bool(os.getenv('OPENAI_API_KEY')),'langsmith_configured':bool(os.getenv('LANGSMITH_API_KEY')),'single_user':True}
    @app.post('/api/offline/compiler')
    async def compiler_offline(data:OfflineDraftDemo,actor=Depends(user)):return await compiler_demo.run(store,data.agent_id,actor,data.count,data.scenario!='success')
    @app.get('/api/compiled-generations')
    def compiled_generations(actor=Depends(user)):return store.all('compiled_generation')
    @app.get('/api/draft-requests')
    def draft_requests(actor=Depends(user)):return [draft_workflow.summarize(j) for j in store.all('draft_request') if j['actor']==actor]
    @app.post('/api/offline/drafts')
    async def offline_drafts(data:OfflineDraftDemo,actor=Depends(user)):
        return await offline_demo.run(store,data.agent_id,actor,data.count,data.scenario)
    @app.get('/api/offline/results')
    def offline_results(actor=Depends(user)):return store.all('offline_demo')
    @app.get('/api/offline/judge-contract')
    def offline_judge_contract(actor=Depends(user)):
        g=store.get('golden','golden-missing_evidence')
        result=judge_contract.fixture(TestCase.model_validate(g['test_case']),Evidence.model_validate(g['evidence']))
        store.put('offline_judge_contract',{'id':uid(),'at':now(),**result});return result
    @app.get('/api/provider/status')
    def provider_status(actor=Depends(user)):return provider.status(store)
    @app.get('/api/provider/plans')
    def provider_plans(actor=Depends(user)):return {'plans':store.all('provider_plan'),'approvals':store.all('provider_approval')}
    @app.post('/api/provider/plans')
    def provider_propose(data:ProviderPlanRequest,actor=Depends(user)):
        return provider.propose_minimal(store,actor,data.agent_id,{'minimal':'all','minimal_generation':'generation','minimal_judge':'judge'}[data.kind]) if data.kind.startswith('minimal') else provider.propose(store,actor,data.agent_id,data.trace,data.retry_of)
    @app.get('/api/provider/plans/{id}/export')
    def provider_export(id:str,actor=Depends(user)):
        p=store.get('provider_plan',id)
        if p['owner']!=actor:raise PolicyError('Provider plan belongs to another owner')
        return JSONResponse(provider.reviewed_plan_export(p),headers={'Content-Disposition':f'attachment; filename=provider-plan-{id}.json'})
    @app.post('/api/provider/plans/{id}/approve')
    def provider_approve(id:str,data:ProviderApprovalRequest,actor=Depends(user)):return provider.approve(store,actor,id,data.digest)
    @app.post('/api/provider/plans/{id}/stop')
    def provider_stop(id:str,actor=Depends(user)):return provider.stop_batch(store,actor,id,'User requested stop')
    @app.post('/api/provider/plans/{id}/connectivity')
    async def provider_connectivity(id:str,actor=Depends(user)):
        with provider.scope(store,actor,id,'connectivity',{'message':'Reply with OK.'}):
            try:
                response=await llm.model(max_tokens=64).ainvoke('Reply with OK.')
                item={'id':uid(),'plan_id':id,'status':'PASS','model':provider.ROLE_MODELS['copilot'],'usage':response.usage_metadata}
                store.put('provider_check',item);return item
            except Exception as exc:raise PolicyError(provider.safe_error(exc)) from None
    @app.get('/api/hr-live/preparations')
    def hr_live_preparations(actor=Depends(user)):return store.all('hr_live_preparation')
    @app.post('/api/hr-live/preparations')
    def prepare_hr_live(actor=Depends(user)):
        from .hr_live_preparation import prepare
        return prepare(store,actor)
    @app.get('/api/hr-live/preparations/{id}')
    def review_hr_live(id:str,actor=Depends(user)):
        from .hr_live_preparation import verify
        return verify(store.get('hr_live_preparation',id))
    @app.get('/api/hr-live/preparations/{id}/export')
    def export_hr_live(id:str,actor=Depends(user)):
        from .hr_live_preparation import verify
        p=verify(store.get('hr_live_preparation',id))
        return JSONResponse(p,headers={'Content-Disposition':'attachment; filename=HR_LIVE_ACCEPTANCE_PLAN_EXACT.json'})
    @app.post('/api/hr-live/preparations/{id}/approve')
    def deny_blocked_hr_live(id:str,actor=Depends(user)):
        from .hr_live_preparation import verify
        verify(store.get('hr_live_preparation',id))
        raise PolicyError('BLOCKED: preparation cannot authorize execution. Resolve isolation/runtime/budget blockers and create a fresh executable plan for human UI approval.')
    @app.get('/api/langsmith/trace-analysis')
    def trace_analysis(actor=Depends(user)):
        from .trace_analysis import analyze
        imports=store.all('langsmith_trace_import')
        if not imports:raise PolicyError('No approved historical trace import exists')
        return analyze(store,imports[0]['evidence_id'])
    @app.get('/api/langsmith/project-listings')
    def remote_project_listings(actor=Depends(user)):return store.all('langsmith_project_listing')
    @app.post('/api/langsmith/prepare-hr-trace-read')
    def prepare_hr_trace_read(actor=Depends(user)):
        g=store.get('golden','golden-hr-sara-authorization')
        selected=next((p for row in store.all('langsmith_project_listing') for p in row['projects'] if p['id']=='22222222-2222-4222-8222-222222222222'),None)
        if not selected:raise PolicyError('Verify the historical HR project through approved project listing first')
        return remote_approval.propose(store,actor,'trace_read',{'project_id':selected['id'],'workspace_id':selected['workspace_id'],'run_id':g['evidence']['trace_id']})
    @app.post('/api/langsmith/trace-read/{id}')
    def retrieve_hr_trace(id:str,actor=Depends(user)):
        from .trace_remote import retrieve
        p=store.get('remote_plan',id)
        with remote_approval.scope(store,actor,id,'trace_read',p['payload']):return retrieve(store,actor,p)
    @app.get('/api/langsmith/plans')
    def remote_plans(actor=Depends(user)):return {'plans':store.all('remote_plan'),'approvals':store.all('remote_approval')}
    @app.get('/api/langsmith/plans/{id}')
    def remote_plan(id:str,actor=Depends(user)):return store.get('remote_plan',id)
    @app.post('/api/langsmith/plans')
    def remote_propose(data:RemotePlanRequest,actor=Depends(user)):return remote_approval.propose(store,actor,data.action,data.payload)
    @app.post('/api/langsmith/plans/{id}/approve')
    def remote_approve(id:str,data:RemoteApprovalRequest,actor=Depends(user)):return remote_approval.approve(store,actor,id,data.digest)
    @app.post('/api/langsmith/selected')
    def selected_import(data:SelectedImport,actor=Depends(user)):
        with remote_approval.scope(store,actor,data.remote_plan_id,'selected',{'run_ids':data.run_ids}):return smith.import_selected(data.run_ids,store)
    @app.get('/api/agents')
    def agents(actor=Depends(user)): return store.all('agent')
    @app.post('/api/agents')
    def add_agent(data:Agent,actor=Depends(user)):
        if data.auth_env and not data.auth_env.startswith('AGENTQA_TARGET_'): raise ValueError('Target secrets must use AGENTQA_TARGET_* environment variables')
        if data.adapter in ('rest','langgraph'): URLPolicy().validate(data.endpoint)
        if data.adapter=='source': raise ValueError('Source references are server-configured only')
        if any(a['id']==data.id for a in store.all('agent')): raise ValueError('Agent ID already exists')
        data.isolation_verified=data.adapter=='demo'; data.profile=None; data.last_discovery=None; data.connection_status='unverified'
        store.put('agent',data); store.audit(actor,'register_agent',{'id':data.id,'adapter':data.adapter})
        return data
    @app.get('/api/agents/{id}/execution-capabilities')
    def mode_capabilities(id:str,actor=Depends(user)):return execution_capabilities(agent(id))
    @app.get('/api/suites/{id}/quality')
    def suite_quality(id:str,actor=Depends(user)):
        s=suite(id);profile=agent(s.agent_id).profile
        return {'suite_id':id,'cases':[{'test_case_id':c.id,**quality_review(c,profile)} for c in s.cases],'note':'Automated review findings are not human approval or verified behavioral success'}
    @app.post('/api/agents/{id}/discover')
    async def discover(id:str,actor=Depends(user)):
        a=agent(id); a.profile=await adapter_for(a).discover(); a.last_discovery=now(); a.connection_status='metadata discovered; business behavior unverified'
        store.put('agent',a); return a
    @app.post('/api/agents/{id}/verify-isolation')
    async def verify_isolation(id:str,actor=Depends(user)):
        a=agent(id)
        if a.adapter not in ('rest','langgraph') or a.environment!='isolated':raise PolicyError('Only isolated HTTP targets support server metadata verification')
        from urllib.parse import urlsplit
        from .adapters import RESTAdapter
        origin=urlsplit(a.endpoint).scheme+'://'+urlsplit(a.endpoint).netloc
        deployments=json.loads(os.getenv('AGENTQA_ISOLATED_DEPLOYMENTS','{}'))
        expected_id=deployments.get(origin)
        if not expected_id:raise PolicyError('Operator must independently verify isolation and configure the expected deployment ID server-side')
        response=await RESTAdapter(a).request('GET','/.well-known/agentqa-isolation')
        response.raise_for_status(); metadata=response.json()
        if metadata.get('deployment_id')!=expected_id or metadata.get('environment')!='isolated' or metadata.get('fictional_data') is not True:raise PolicyError('Isolation metadata does not match trusted deployment configuration')
        a.isolation_verified=True;store.put('agent',a)
        store.audit(actor,'verify_isolation',{'agent_id':id,'deployment_id':expected_id,'metadata':redact(metadata)})
        return a
    @app.put('/api/agents/{id}/profile')
    def enrich(id:str,data:AgentProfile,actor=Depends(user)):
        a=agent(id)
        for c in data.capabilities:
            c.provenance='user_declared'; c.evidence.append('Authenticated manual profile enrichment')
        data.fingerprint=fingerprint(data); a.profile=data; store.put('agent',a)
        store.audit(actor,'enrich_profile',{'agent_id':id,'fingerprint':data.fingerprint}); return a
    @app.get('/api/suites')
    def suites(actor=Depends(user)): return store.all('suite')
    @app.post('/api/agents/{id}/generate')
    async def generate_suite(id:str,data:GenerateRequest,actor=Depends(user)):
        a=agent(id)
        if data.ai:
            with provider.scope(store,actor,data.provider_plan_id,'generation',{'agent':a.model_dump(mode='json',by_alias=True),'count':data.count,'focus':data.focus}):
                s=await llm.ai_generate(a,data.count,data.focus,data.budget,store,actor)
        else:s=generate(a,data.count,data.focus)
        store.put('suite',s); store.audit(actor,'generate_suite',{'suite_id':s.id,'count':len(s.cases),'source':s.generation}); return s
    @app.put('/api/suites/{id}')
    def edit_suite(id:str,data:Suite,actor=Depends(user)):
        old=suite(id)
        if data.id!=id or data.agent_id!=old.agent_id: raise ValueError('Suite identity cannot change')
        a=agent(data.agent_id)
        for c in data.cases: validate_case(c,a.profile)
        if len({c.id for c in data.cases})!=len(data.cases): raise ValueError('Duplicate test IDs')
        store.put('suite',data); store.audit(actor,'edit_suite',{'suite_id':id}); return data
    @app.get('/api/agents/{id}/coverage')
    def get_coverage(id:str,actor=Depends(user)): return coverage(agent(id),store.all('suite'),store.all('run'))
    @app.get('/api/suites/{id}/recommend')
    def recommend(id:str,budget:float=5,actor=Depends(user)): return prioritize(suite(id),budget,store.all('run'))
    @app.post('/api/plans')
    def propose(data:PlanRequest,actor=Depends(user)):
        options=data.model_dump(exclude={'agent_id','suite_id'})
        return policy.propose(agent(data.agent_id),suite(data.suite_id),**options)
    @app.get('/api/plans')
    def plans(actor=Depends(user)): return store.all('plan')
    @app.post('/api/plans/{id}/approve')
    def approve(id:str,data:ApprovalRequest,actor=Depends(user)):
        return policy.approve(id,actor,data.digest,data.additional_authorization)
    @app.post('/api/plans/{id}/run')
    async def start(id:str,actor=Depends(user)): return runner.start(id,actor)
    @app.get('/api/runs')
    def runs(actor=Depends(user)): return store.all('run')
    @app.get('/api/runs/{id}')
    def get_run(id:str,actor=Depends(user)): return store.get('run',id)
    @app.get('/api/runs/{id}/partials')
    def partials(id:str,actor=Depends(user)):
        store.get('run',id)
        return [p for p in store.all('partial') if p['run_id']==id]
    @app.post('/api/runs/{id}/cancel')
    def cancel(id:str,actor=Depends(user)): return runner.cancel(id,actor)
    @app.get('/api/runs/{id}/failures')
    def failures(id:str,actor=Depends(user)):
        r=store.get('run',id);return clusters(r,store.get('plan',r['plan_id']))
    @app.post('/api/runs/{id}/override')
    def override(id:str,data:OverrideRequest,actor=Depends(user)):
        r=store.get('run',id)
        result=next((x for x in r['results'] if x['id']==data.result_id),None)
        if not result: raise ValueError('Unknown result')
        item={'id':uid(),'run_id':id,'actor':actor,'at':now(),**data.model_dump()}
        store.put('override',item); store.audit(actor,'manual_override',item)
        return {'override':item,'original_preserved':True}
    @app.get('/api/overrides')
    def overrides(actor=Depends(user)): return store.all('override')
    @app.get('/api/compare')
    def comparison(before:str,after:str,actor=Depends(user)): return compare(store.get('run',before),store.get('run',after))
    @app.get('/api/reports/{id}')
    def get_report(id:str,format:Literal['json','md','html']='json',actor=Depends(user)):
        data=report(store,id)
        body=json.dumps(data,indent=2,ensure_ascii=False) if format=='json' else markdown_report(data) if format=='md' else html_report(data)
        return Response(body,media_type={'json':'application/json','md':'text/markdown','html':'text/html'}[format],headers={'Content-Disposition':f'attachment; filename="agentqa-{id}.{format}"'})
    @app.post('/api/chat')
    async def copilot(data:ChatRequest,actor=Depends(user)):
        agent(data.agent_id)
        if data.provider_plan_id:
            with provider.scope(store,actor,data.provider_plan_id,'copilot',{'agent':store.get('agent',data.agent_id),'message':data.message,'session_id':data.session_id}):
                return await llm.chat(store,data.agent_id,actor,data.session_id,data.message,data.budget)
        return await llm.chat(store,data.agent_id,actor,data.session_id,data.message,data.budget)
    @app.get('/api/copilot/evidence')
    def copilot_evidence(actor=Depends(user)):return {'completed_graphs':store.all('copilot_evidence'),'incremental_tool_events':store.all('copilot_tool_events')}
    @app.get('/api/chats')
    def chats(actor=Depends(user)): return [c for c in store.all('chat') if c['owner']==actor]
    @app.get('/api/evidence')
    def evidence(actor=Depends(user)): return store.all('evidence')
    @app.post('/api/evidence')
    def import_evidence(data:Evidence,actor=Depends(user)):
        if any(e['id']==data.id for e in store.all('evidence')):raise ValueError('Evidence ID already exists; import a new record rather than overwrite history')
        data.source='historical'; data.raw=redact(data.raw); data.tool_calls=redact(data.tool_calls); store.put('evidence',data); store.audit(actor,'import_evidence',{'id':data.id}); return data
    @app.get('/api/langsmith/projects')
    def projects(remote_plan_id:str|None=None,actor=Depends(user)):
        with remote_approval.scope(store,actor,remote_plan_id,'projects',{}):return smith.projects()
    @app.post('/api/langsmith/import')
    def import_traces(data:TraceImport,actor=Depends(user)):
        with remote_approval.scope(store,actor,data.remote_plan_id,'import',data.model_dump(exclude={'remote_plan_id'})):result=smith.import_runs(data.project,store,data.limit)
        store.audit(actor,'import_langsmith',result); return result
    @app.post('/api/langsmith/dataset')
    def dataset(data:DatasetRequest,actor=Depends(user)):
        s=suite(data.suite_id)
        with remote_approval.scope(store,actor,data.remote_plan_id,'dataset',data.model_dump(exclude={'remote_plan_id'})):result=smith.create_dataset(data.name,[{'inputs':{'turns':c.turns},'outputs':{'expected_behavior':c.expected_behavior}} for c in s.cases]); store.audit(actor,'create_langsmith_dataset',result); return result
    @app.post('/api/langsmith/feedback')
    def feedback(data:FeedbackRequest,actor=Depends(user)):
        with remote_approval.scope(store,actor,data.remote_plan_id,'feedback',data.model_dump(exclude={'remote_plan_id'})):result=smith.feedback(**data.model_dump(exclude={'remote_plan_id'}))
        store.audit(actor,'langsmith_feedback',result); return result
    @app.post('/api/langsmith/experiment')
    def experiment(data:ExperimentRequest,actor=Depends(user)):
        with remote_approval.scope(store,actor,data.remote_plan_id,'experiment',data.model_dump(exclude={'remote_plan_id'})):result=smith.record_experiment(data.name,report(store,data.run_id)); store.audit(actor,'langsmith_experiment',result); return result
    @app.get('/api/calibration')
    def get_calibration(actor=Depends(user)): return calibration(store)
    @app.post('/api/golden/{id}/judge')
    async def judge_golden(id:str,data:GoldenJudgeRequest,actor=Depends(user)):
        old=store.get('golden',id)
        if data.budget<llm.RESERVATION: raise ValueError('Evaluation budget too small')
        case=TestCase.model_validate(old['test_case']); evidence=Evidence.model_validate(old['evidence'])
        p=store.get('provider_plan',data.provider_plan_id) if data.provider_plan_id else {}
        profile=store.get('agent',p['agent_id']).get('profile') if p else None
        with provider.scope(store,actor,data.provider_plan_id,'judge:'+id,{'case':old['test_case'],'evidence':old['evidence'],'profile':profile}):
            findings=await llm.judge(case,evidence,store,actor,profile=profile)
        from .evaluation import aggregate
        deterministic=__import__('agentqa.evaluation',fromlist=['evaluate']).evaluate(case,evidence)
        updated={**old,'aggregated_verdict':aggregate([f.verdict for f in deterministic]+[f.verdict for f in findings]),'deterministic_findings':[f.model_dump(mode='json') for f in deterministic],'judge_verdict':aggregate([f.verdict for f in findings]),'judge_findings':[f.model_dump(mode='json') for f in findings]}
        store.audit(actor,'calibrate_judge',{'golden_id':id,'findings':updated['judge_findings']}); store.put('golden',updated)
        return updated
    @app.get('/api/golden')
    def golden(actor=Depends(user)): return store.all('golden')
    @app.put('/api/golden/{id}')
    def golden_review(id:str,data:GoldenUpdate,actor=Depends(user)):
        old=store.get('golden',id); updated={**old,**data.model_dump(),'reviewed_by':actor,'reviewed_at':now()}
        store.audit(actor,'golden_review',{'before':old,'after':updated}); store.put('golden',updated); return updated
    @app.get('/api/usage')
    def usage(actor=Depends(user)): return {'llm_calls':store.all('usage'),'runs':[{'id':r['id'],'reserved_cost':r['reserved_cost'],'measured_cost':r['measured_cost'],'calls':r['calls']} for r in store.all('run')]}
    @app.get('/api/audit')
    def audit(actor=Depends(user)): return store.audits()
    static=Path(__file__).parent/'static'
    app.mount('/static',StaticFiles(directory=static),name='static')
    @app.get('/')
    def home(): return FileResponse(static/'index.html')
    return app

app=create_app()
