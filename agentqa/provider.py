"""Explicit human approval for a bounded provider batch, independent of target plans."""
import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from langchain_core.callbacks import BaseCallbackHandler
from .models import now, uid
from .security import PolicyError
from .config import ROLE_MODELS

ACCEPTANCE_PROMPT='Analyze the connected Company/HR Agent. Explain its tools, business operations, argument schemas, conversation capabilities, and observable limitations. Then propose 40 meaningful test cases covering tool selection, argument extraction, memory, authorization, multi-turn workflows, error handling, and response grounding. Prioritize the tests, identify coverage gaps, and estimate costs. Do not execute any live tests.'
CURRENT=ContextVar('provider_approval',default=None)
INPUT_LIMIT=128000
OUTPUT_LIMIT=6000
# This price schedule is valid only for this explicit candidate snapshot.
PRICES={'openai:gpt-5.4-mini-2026-03-17':(.75,4.50)}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def status(store=None):
    verified=[c['model'] for c in store.all('provider_check') if c.get('status')=='PASS'] if store else []
    return {'configured':bool(os.getenv('OPENAI_API_KEY')),'models':ROLE_MODELS,'account_model_access':'VERIFIED by approved connectivity check' if all(m in verified for m in ROLE_MODELS.values()) else 'UNVERIFIED until approved connectivity check','verified_models':verified,'langsmith_configured':bool(os.getenv('LANGSMITH_API_KEY')),'langsmith_project':os.getenv('LANGSMITH_PROJECT','agentqa-copilot'),'tracing_configured':os.getenv('LANGSMITH_TRACING','').lower()=='true','remote_access':'UNVERIFIED; remote requests require separate approval','paid_calls_require':'exact provider plan approval','secrets_returned':False}
def propose(store,actor,agent_id,trace=False,retry_of=None):
    a=store.get('agent',agent_id)
    if a['adapter']!='source':raise PolicyError('Initial acceptance batch requires the source-only reference agent')
    if any(m not in PRICES for m in ROLE_MODELS.values()):raise PolicyError('Configure a priced supported model before preparing a bounded plan')
    records=store.all('golden')
    chosen=['golden-hr-sara-authorization','golden-grounded','golden-missing_evidence']
    golden=[next(g for g in records if g['id']==id) for id in chosen] if all(any(g['id']==id for g in records) for id in chosen) else [g for g in records if g.get('test_case') and g.get('evidence')][:3]
    if len(golden)!=3:raise PolicyError('Three recorded golden examples are required')
    operations=[{'name':'connectivity','role':'copilot','max_calls':1,'binding':digest({'message':'Reply with OK.'})},
      {'name':'copilot','role':'copilot','max_calls':6,'binding':digest({'agent':a,'message':ACCEPTANCE_PROMPT,'session_id':None})},
      {'name':'generation','role':'generation','max_calls':1,'binding':digest({'agent':a,'count':3,'focus':'live AI acceptance'})}]
    for g in golden:operations.append({'name':'judge:'+g['id'],'role':'judge','max_calls':1,'binding':digest({'case':g['test_case'],'evidence':g['evidence'],'profile':a.get('profile')})})
    recovery_exposure=0
    if retry_of:
        previous=store.get('provider_plan',retry_of);approval=store.get('provider_approval',retry_of)
        if previous['owner']!=actor or previous['agent_id']!=agent_id or not approval['stopped']:raise PolicyError('Recovery requires this owner/agent and a stopped prior batch')
        if previous.get('retry_of'):raise PolicyError('Repeated recovery requires a separately designed scope and budget')
        usage=[u for u in store.all('usage') if u.get('plan_id')==retry_of and u['category']=='provider_dispatch']
        recovery_exposure=sum(u.get('reported_token_cost_estimate',u['reserved_cost']) for u in usage)
        operations=[o for o in operations if o['name']!='connectivity']
    for o in operations:
        o['input_limit']=128000 if o['name']=='copilot' else 64000 if o['name']=='generation' else 512 if o['name']=='connectivity' else 32000
        if retry_of and o['name']=='copilot':o['input_limit']=100000
        o['output_limit']=64 if o['name']=='connectivity' else 3000 if o['role']=='judge' else 6000
    maximum=sum(o['max_calls']*(o['input_limit']*PRICES[ROLE_MODELS[o['role']]][0]+o['output_limit']*PRICES[ROLE_MODELS[o['role']]][1])/1e6 for o in operations)
    p={'id':uid(),'created_at':now(),'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),'owner':actor,'agent_id':agent_id,'agent_snapshot_digest':digest(a),'models':dict(ROLE_MODELS),'operations':operations,'platform_tools':['agent_profile','generate_tests','save_proposed_tests','coverage_gaps','valuable_tests'],'max_calls':sum(o['max_calls'] for o in operations),'max_input_tokens_per_call':INPUT_LIMIT,'max_output_tokens_per_call':OUTPUT_LIMIT,'maximum_estimated_cost_usd':maximum,'budget_usd':min(0.8,round(1.0-recovery_exposure,6)) if retry_of else 1.0,'retry_of':retry_of,'prior_batch_conservative_exposure_usd':recovery_exposure,'prices_per_million':PRICES,'prompt':ACCEPTANCE_PROMPT,'generation':{'count':3,'focus':'live AI acceptance'},'golden_ids':[g['id'] for g in golden],'trace':trace,'disclosure':'Sends selected HR source metadata, exact prompt, and three recorded examples to OpenAI. No target execution. No LangSmith export or tracing. Application ceiling, not an account billing cap.'}
    if maximum>p['budget_usd']:raise PolicyError('Estimated maximum plus prior batch exposure exceeds $1; reduce scope')
    if trace:
        from . import remote_approval
        remote=remote_approval.propose(store,actor,'tracing',{'provider_plan_id':p['id'],'project':os.getenv('LANGSMITH_PROJECT','agentqa-copilot')})
        p['remote_trace_plan_id']=remote['id'];p['disclosure']='Sends this approved batch to OpenAI and its model inputs/outputs to the separately approved LangSmith project. No target execution.'
    p['digest']=digest(p);store.put('provider_plan',p);store.audit(actor,'provider_plan_proposed',{'id':p['id'],'digest':p['digest']});return p
def approve(store,actor,id,approval_digest):
    p=store.get('provider_plan',id)
    if p['owner']!=actor or p['digest']!=approval_digest or datetime.fromisoformat(p['expires_at'])<datetime.now(timezone.utc):raise PolicyError('Provider scope/digest/owner/expiry mismatch')
    if store.all('provider_approval') and any(a['id']==id for a in store.all('provider_approval')):raise PolicyError('Provider approval already exists')
    record={'id':id,'actor':actor,'digest':p['digest'],'at':now(),'calls':{},'reserved_cost':0,'reported_cost_estimate':0,'stopped':False}
    import sqlite3
    try:
        with store.lock,store.connect() as c:c.execute('INSERT INTO documents VALUES(?,?,?)',('provider_approval',id,json.dumps(record)))
    except sqlite3.IntegrityError:raise PolicyError('Provider approval already exists; counters cannot be reset') from None
    store.audit(actor,'approve_provider_plan',record);return record
def stop_batch(store,actor,plan_id,reason):
    p=store.get('provider_plan',plan_id)
    if p['owner']!=actor:raise PolicyError('Provider plan belongs to another owner')
    with store.lock,store.connect() as c:
        c.execute('BEGIN IMMEDIATE');row=c.execute("SELECT body FROM documents WHERE kind='provider_approval' AND id=?",(plan_id,)).fetchone()
        if not row:raise PolicyError('Provider plan is not approved')
        a=json.loads(row['body']);a['stopped']=True;a['stop_reason']=reason;a['stopped_at']=now()
        c.execute("UPDATE documents SET body=? WHERE kind='provider_approval' AND id=?",(json.dumps(a),plan_id))
    store.audit(actor,'stop_provider_batch',{'plan_id':plan_id,'reason':reason})
    return a

@contextmanager
def scope(store,actor,plan_id,operation,binding):
    if not plan_id:raise PolicyError('Fresh explicit provider approval required; no paid request dispatched')
    p=store.get('provider_plan',plan_id)
    try:a=store.get('provider_approval',plan_id)
    except KeyError:raise PolicyError('This exact provider plan has not been explicitly approved; no request dispatched') from None
    op=next((o for o in p['operations'] if o['name']==operation),None)
    unsigned={k:v for k,v in p.items() if k!='digest'}
    if not op or digest(unsigned)!=p['digest'] or a['digest']!=p['digest'] or p['owner']!=actor or a['actor']!=actor or op['binding']!=digest(binding) or p['models']!=ROLE_MODELS or datetime.fromisoformat(p['expires_at'])<datetime.now(timezone.utc):raise PolicyError('Provider approval does not match exact operation, payload, owner, model or expiry')
    if p.get('kind')=='minimal' and digest(implementation_manifest(store,p['agent_id'],actor))!=p['implementation_digest']:raise PolicyError('Prompt or tool/output schema changed; create and review a fresh plan')
    if a['stopped']:raise PolicyError('Provider batch is stopped; no request dispatched')
    if not os.getenv('OPENAI_API_KEY'):raise PolicyError('OPENAI_API_KEY missing. Set it privately in the ignored server .env and restart; no request dispatched')
    token=CURRENT.set({'store':store,'actor':actor,'plan':p,'operation':op,'invocation_id':uid()})
    from langsmith import tracing_context
    try:
        if p.get('trace'):
            from . import remote_approval,langsmith_integration
            payload={'provider_plan_id':p['id'],'project':os.getenv('LANGSMITH_PROJECT','agentqa-copilot')}
            with remote_approval.scope(store,actor,p['remote_trace_plan_id'],'tracing',payload):
                client=langsmith_integration.client()
                with tracing_context(enabled=True,client=client,project_name=payload['project']):yield
                client.flush();client.close()
        else:
            with tracing_context(enabled=False):yield
    except BaseException as error:
        stop_batch(store,actor,plan_id,'Approved operation failed: '+type(error).__name__)
        raise
    finally:CURRENT.reset(token)
class ProviderGuard(BaseCallbackHandler):
    raise_error=True
    def __init__(self,context):self.ctx=context;self.records={}
    def on_chat_model_start(self,serialized,messages,*,run_id,**kwargs):
        ctx=self.ctx;p=ctx['plan'];op=ctx['operation'];store=ctx['store']
        fresh=store.get('provider_plan',p['id'])
        if fresh!=p or digest(store.get('agent',p['agent_id']))!=p['agent_snapshot_digest'] or datetime.fromisoformat(p['expires_at'])<datetime.now(timezone.utc):raise PolicyError('Provider plan expired or agent changed before dispatch')
        # UTF-8 byte bound is deliberately conservative; includes allowance for schemas/serialization.
        bound=len(str(messages).encode('utf-8'))+(64 if op['name']=='connectivity' else 12000)
        if bound>op['input_limit']:raise PolicyError('Input exceeds conservative token bound; no provider dispatch')
        cost=(op['input_limit']*PRICES[p['models'][op['role']]][0]+op['output_limit']*PRICES[p['models'][op['role']]][1])/1e6
        with store.lock,store.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute("SELECT body FROM documents WHERE kind='provider_approval' AND id=?",(p['id'],)).fetchone();a=json.loads(row['body'])
            invocations=a.setdefault('invocations',{})
            if op['name'] in invocations and invocations[op['name']]!=ctx['invocation_id']:raise PolicyError('This operation was already started; obtain fresh approval to retry')
            invocations[op['name']]=ctx['invocation_id']
            calls=a['calls'].get(op['name'],0)
            if a['stopped'] or calls>=op['max_calls'] or a['reserved_cost']+cost>p['budget_usd']:raise PolicyError('Provider call/cost ceiling exhausted or batch stopped')
            a['calls'][op['name']]=calls+1;a['reserved_cost']+=cost
            c.execute("UPDATE documents SET body=? WHERE kind='provider_approval' AND id=?",(json.dumps(a),p['id']))
        item={'id':uid(),'at':now(),'actor':ctx['actor'],'category':'provider_dispatch','operation':op['name'],'plan_id':p['id'],'model':p['models'][op['role']],'reserved_cost':cost,'measured_cost':None,'usage':{},'status':'dispatched','trace_delivery':'UNVERIFIED' if p.get('trace') else 'disabled','trace_id':str(run_id) if p.get('trace') else None,'trace_url':('https://smith.langchain.com/r/'+str(run_id)) if p.get('trace') else None}
        self.records[str(run_id)]=item;store.put('usage',item);store.audit(ctx['actor'],'provider_dispatch_authorized',{'plan_id':p['id'],'operation':op['name'],'usage_id':item['id']})
    def on_llm_end(self,response,*,run_id,**kwargs):
        item=self.records.get(str(run_id))
        if not item:return
        usage=getattr(response.generations[0][0].message,'usage_metadata',None) or {}
        inp=usage.get('input_tokens');out=usage.get('output_tokens');unexpected=type(inp) is not int or type(out) is not int or inp<0 or out<0 or inp>self.ctx['operation']['input_limit'] or out>self.ctx['operation']['output_limit']
        item['finish_reason']=getattr(response.generations[0][0].message,'response_metadata',{}).get('finish_reason')
        item['provider_reported_model']=getattr(response.generations[0][0].message,'response_metadata',{}).get('model_name')
        item['usage']=usage;item['status']='stopped_unexpected_usage' if unexpected else 'completed'
        if not unexpected:
            prices=PRICES[item['model']];item['reported_token_cost_estimate']=(inp*prices[0]+out*prices[1])/1e6
            store=self.ctx['store']
            with store.lock,store.connect() as c:
                c.execute('BEGIN IMMEDIATE');row=c.execute("SELECT body FROM documents WHERE kind='provider_approval' AND id=?",(self.ctx['plan']['id'],)).fetchone();a=json.loads(row['body'])
                a['reported_cost_estimate']+=item['reported_token_cost_estimate']
                c.execute("UPDATE documents SET body=? WHERE kind='provider_approval' AND id=?",(json.dumps(a),a['id']))
        self.ctx['store'].put('usage',item)
        if unexpected:
            self.stop();raise PolicyError('Unexpected or absent provider usage; batch stopped for review')
    def stop(self):
        store=self.ctx['store']
        with store.lock,store.connect() as c:
            c.execute('BEGIN IMMEDIATE');row=c.execute("SELECT body FROM documents WHERE kind='provider_approval' AND id=?",(self.ctx['plan']['id'],)).fetchone();a=json.loads(row['body']);a['stopped']=True
            c.execute("UPDATE documents SET body=? WHERE kind='provider_approval' AND id=?",(json.dumps(a),a['id']))
    def on_llm_error(self,error,*,run_id,**kwargs):
        item=self.records.get(str(run_id))
        if item:
            item['status']='provider_error';item['diagnostic']=safe_error(error);item['error_type']=type(error).__name__;item['http_status']=getattr(error,'status_code',None);self.ctx['store'].put('usage',item)
        self.stop()
def safe_error(error):
    name=type(error).__name__
    if 'Authentication' in name:return 'Provider authentication rejected. Replace the private key and prepare a fresh approval.'
    if 'NotFound' in name or 'Permission' in name:return 'Configured model unavailable or account permission denied. Verify model access; prepare a fresh plan.'
    if 'RateLimit' in name:return 'Provider quota/rate limit rejected the request. Check account billing/quota privately.'
    if 'InvalidRequest' in name or 'BadRequest' in name:return 'Provider rejected the request format or parameters. Inspect the safe schema diagnostic and prepare a fresh approval after correction.'
    return 'Provider request failed; batch stopped. Check server configuration and connectivity privately.'


def implementation_manifest(store,agent_id,actor):
    from . import llm,draft_workflow
    from .models import JudgeOutput
    from .models import Agent
    scoped=Agent.model_validate(store.get('agent',agent_id));scoped.profile.capabilities=[c for c in scoped.profile.capabilities if c.name=='get_leave_balance']
    names={'agent_profile','generate_tests'}
    from pathlib import Path
    code_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('llm.py','draft_workflow.py','judge_contract.py','generation.py','provider.py','scenario_compiler.py')}
    return {'code_hashes':code_hashes,'copilot_system':llm.GROUNDED_SYSTEM,'generation_system':llm.SCENARIO_SYSTEM,'judge_system':llm.JUDGE_SYSTEM,'rubric_version':llm.RUBRIC_VERSION,'tool_schemas':{t.name:t.args_schema.model_json_schema() for t in llm.grounded_copilot_tools(store,agent_id,actor) if t.name in names},'generation_schema':__import__('agentqa.scenario_compiler',fromlist=['proposal_schema']).proposal_schema(scoped),'judge_schema':JudgeOutput.model_json_schema(),'structured_output':{'generation':{'method':'function_calling','strict':False,'application_validation':'per-proposal intent validation and deterministic contract compiler'},'judge':{'method':'json_schema','strict':True,'application_validation':'strict JSON parse, all ten criteria, real evidence refs'}}}


def propose_minimal(store,actor,agent_id,flow="all"):
    """Prepare only. Requires a new authenticated human approval before any provider request."""
    from . import llm
    from .models import Agent,TestCase,Evidence
    a=store.get('agent',agent_id)
    if a['adapter']!='source':raise PolicyError('Minimal acceptance requires the source-only reference; no target invocation')
    if any(m not in PRICES for m in ROLE_MODELS.values()):raise PolicyError('No reviewed price schedule for configured model')
    if not any(c['name']=='get_leave_balance' for c in a.get('profile',{}).get('capabilities',[])):raise PolicyError('get_leave_balance source contract required')
    prompt='Generate exactly 3 distinct structured draft test cases for get_leave_balance: omitted optional year, explicit null year, and missing required employee identity without trusted identity. Use generate_tests for the selected agent with count=3; the backend must compile intent and persist valid drafts, and report verified counts and limitations. Do not execute tests.'
    focus='Three get_leave_balance cases: omitted optional year, explicit null year, missing employee ID without trusted identity; unreviewed drafts only.'
    g=store.get('golden','golden-hr-sara-authorization')
    operations=[{'name':'copilot','role':'copilot','max_calls':4,'input_limit':29000,'output_limit':2900,'binding':digest({'agent':a,'message':prompt,'session_id':None})}, {'name':'generation','role':'generation','max_calls':1,'input_limit':24000,'output_limit':2500,'binding':digest({'agent':a,'count':3,'focus':focus})}, {'name':'judge:'+g['id'],'role':'judge','max_calls':1,'input_limit':24000,'output_limit':3000,'binding':digest({'case':g['test_case'],'evidence':g['evidence'],'profile':a.get('profile')})}]
    maximum=sum(o['max_calls']*(o['input_limit']*PRICES[ROLE_MODELS[o['role']]][0]+o['output_limit']*PRICES[ROLE_MODELS[o['role']]][1])/1e6 for o in operations)
    manifest=implementation_manifest(store,agent_id,actor)
    scoped=Agent.model_validate(a);scoped.profile.capabilities=[c for c in scoped.profile.capabilities if c.name=='get_leave_balance'];scoped.profile.tools=sorted({c.tool for c in scoped.profile.capabilities})
    p={'id':uid(),'kind':'minimal','created_at':now(),'expires_at':(datetime.now(timezone.utc)+timedelta(hours=24)).isoformat(),'owner':actor,'agent_id':agent_id,'agent_snapshot_digest':digest(a),'models':dict(ROLE_MODELS),'operations':operations,'platform_tools':['agent_profile','generate_tests'],'capability_scope':['get_leave_balance'],'max_calls':6,'maximum_estimated_cost_usd':maximum,'budget_usd':.20,'prices_per_million':PRICES,'prompt':prompt,'generation':{'count':3,'focus':focus},'golden_ids':[g['id']],'trace':False,'implementation_manifest':manifest,'implementation_digest':digest(manifest),'reviewed_data':{'scoped_agent_profile':scoped.model_dump(mode='json',by_alias=True),'judge_case':g['test_case'],'judge_evidence':g['evidence'],'judge_contracts':llm.relevant_contracts(a['profile'],TestCase.model_validate(g['test_case']),Evidence.model_validate(g['evidence']))},'disclosure':'OpenAI only: selected get_leave_balance source contract, exact prompts, three draft cases per interaction/direct generation, and the recorded Sara backend-denial example with relevant source contracts. No target calls, no LangSmith requests or tracing. At most six model calls; no implicit retries or refills. Existing stopped approvals remain stopped.'}
    if maximum>.20:raise PolicyError('Minimal plan cannot fit $0.20')
    if flow not in ('all','generation','judge'):raise PolicyError('Unknown acceptance flow')
    if flow!='all':
        p['acceptance_flow']=flow
        p['operations']=[o for o in operations if (o['role']=='judge')==(flow=='judge')]
        p['max_calls']=sum(o['max_calls'] for o in p['operations'])
        p['maximum_estimated_cost_usd']=sum(o['max_calls']*(o['input_limit']*PRICES[ROLE_MODELS[o['role']]][0]+o['output_limit']*PRICES[ROLE_MODELS[o['role']]][1])/1e6 for o in p['operations'])
        p['budget_usd']=round(p['maximum_estimated_cost_usd'],8)
        if flow=='generation':
            p['golden_ids']=[]
            for key in ('judge_case','judge_evidence','judge_contracts'):p['reviewed_data'].pop(key,None)
        else:
            p['platform_tools']=[];p['prompt']='';p['generation']={}
        p['disclosure']+=' This immutable approval covers only the '+flow+' flow. The other flow requires its own approval and has independent stop counters.'
    p['digest']=digest(p);store.put('provider_plan',p);store.audit(actor,'provider_plan_proposed',{'id':p['id'],'digest':p['digest'],'kind':'minimal'});return p


def reviewed_plan_export(plan):
    """Preserve immutable data exactly. Reject credentials; never redact business policy fields."""
    if digest({k:v for k,v in plan.items() if k!='digest'})!=plan['digest']:
        raise PolicyError('Plan digest does not match; export blocked')
    serialized=json.dumps(plan,ensure_ascii=False,allow_nan=False)
    for name in ('OPENAI_API_KEY','LANGSMITH_API_KEY','AGENTQA_ADMIN_TOKEN'):
        secret=os.getenv(name)
        if secret and secret in serialized:raise PolicyError('Credential detected in plan data; export blocked. Remove it privately and prepare a new plan.')
    return plan
