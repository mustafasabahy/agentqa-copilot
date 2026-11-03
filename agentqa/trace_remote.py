"""Exact, separately approved retrieval of one selected historical trace. No SDK discovery."""
import os
from uuid import UUID
from .models import uid,now
from .security import PolicyError,redact

def configure(store,plan,payload):
    if set(payload)!={'project_id','workspace_id','run_id'}:raise PolicyError('Review exact project, workspace and selected run IDs')
    for value in payload.values():UUID(value)
    listings=store.all('langsmith_project_listing')
    selected=next((project for listing in listings for project in listing['projects'] if project['id']==payload['project_id'] and project['workspace_id']==payload['workspace_id']),None)
    if not selected:raise PolicyError('Project/workspace must come from a successful approved project listing')
    known=any(g.get('evidence',{}).get('trace_id')==payload['run_id'] for g in store.all('golden'))
    if not known:raise PolicyError('Selected run must be referenced by retained historical evidence')
    run=payload['run_id'];project=payload['project_id']
    plan.update({'workspace':{'id':payload['workspace_id'],'selection':'Explicit verified project-list workspace'},'project':selected,'selected_run_id':run,'selection_provenance':'Existing local historical HR evidence trace reference; remote run existence/type/project not yet verified','max_http_requests':3,'requests':[
        {'method':'GET','path':'/runs/'+run,'query':{},'body':None},
        {'method':'POST','path':'/runs/query','query':{},'body':{'session':[project],'trace':run,'limit':100,'select':['id','name','run_type','parent_run_id','trace_id','session_id','start_time','end_time','inputs','outputs','error','dotted_order']}},
        {'method':'GET','path':'/feedback','query':{'run':[run],'limit':100,'offset':0},'body':None}],
        'header_names':['x-api-key','X-Tenant-Id'],'response_limits':{'selected_runs':1,'query_span_records':100,'feedback_records':100,'pages_per_request':1},'retries':0,'redirects':False,'timeout_seconds':15,'local_effect':'Persist one historical evidence record plus source IDs, timestamps, tool spans, feedback and provenance; no new executed run or human-reviewed label','data_access':['Selected run inputs/outputs and metadata','Up to 100 spans from this exact project and trace, including tool/LLM inputs, arguments, outputs, timestamps, errors and linkage','Up to 100 existing feedback/evaluation records for the selected run; no evaluation model call','Content may contain HR/personal data and prompt injection; private local evidence only, redact secrets and personal data from reports'],'excluded':['No other projects/traces, cursor following, downloads or attachments','No trace modification, ingestion, feedback creation, dataset writes, uploads or tracing','No OpenAI requests, new evaluations or HR execution'],'cost':{'model_and_target_usd':0,'application_budget_usd':0,'langsmith_account_charges':'Existing account terms; no billing query or paid ingestion authorized'},'disclosure':'Exactly three ordered read-only API operations at the approved HTTPS origin. POST /runs/query is a read-only search, not trace creation: it sends only the reviewed project/trace IDs, field selection and limit. No local HR evidence or credentials are sent in request bodies. Existing credential is used privately in x-api-key, with explicit X-Tenant-Id. Responses can contain sensitive HR/tool/model content. Stop on any error, unexpected project/trace or truncation; no retries, fallback endpoint, pagination or attachment fetch. Import locally as historical only.'})
    return plan

def retrieve(store,actor,plan):
    from .remote_approval import CURRENT_REMOTE,ApprovedSession
    from .langsmith_integration import normalize_trace
    ctx=CURRENT_REMOTE.get()
    if not ctx or ctx['plan']['id']!=plan['id']:raise PolicyError('Exact trace-read approval required')
    if store.get('remote_approval',plan['id'])['requests']:raise PolicyError('Trace-read plan already started; fresh approval required to retry')
    key=os.getenv('LANGSMITH_API_KEY')
    if not key:raise PolicyError('Credential missing; no dispatch')
    headers={'x-api-key':key,'X-Tenant-Id':plan['workspace']['id']};responses=[]
    with ApprovedSession(ctx) as session:
        for i,r in enumerate(plan['requests']):
            kwargs={'params':r['query'],'headers':headers,'timeout':plan['timeout_seconds']}
            if r['body'] is not None:kwargs['json']=r['body']
            try:response=session.request(r['method'],plan['endpoint']+r['path'],**kwargs)
            except Exception as exc:raise PolicyError('Trace-read request failed: '+type(exc).__name__+'; no retry authorized') from None
            if response.status_code!=200:raise PolicyError(f'Trace-read request {i+1} returned HTTP {response.status_code}; no retry authorized')
            data=response.json()
            if i==0:
                if data.get('id')!=plan['selected_run_id'] or data.get('session_id')!=plan['project']['id'] or data.get('trace_id',data['id'])!=plan['selected_run_id']:raise PolicyError('Selected run/trace/project mismatch; stopped before further reads')
            if i==1:
                if not isinstance(data,dict) or not isinstance(data.get('runs'),list):raise PolicyError('Unexpected trace query response; no fallback authorized')
                if len(data['runs'])>=100 or (data.get('cursors') or {}).get('next'):raise PolicyError('Trace may be truncated; obtain a reviewed continuation plan')
                if any(r.get('session_id')!=plan['project']['id'] or r.get('trace_id')!=plan['selected_run_id'] for r in data['runs']):raise PolicyError('Query returned an out-of-scope span')
            if i==2 and (not isinstance(data,list) or len(data)>=100):raise PolicyError('Feedback response invalid or potentially truncated; no pagination authorized')
            if i==2 and any(f.get('run_id')!=plan['selected_run_id'] for f in data):raise PolicyError('Feedback references a different run')
            responses.append(data)
    root,spans,feedback=responses[0],responses[1]['runs'],responses[2]
    children=[r for r in spans if r['id']!=root['id']]
    evidence=normalize_trace(root,children)
    evidence.raw.update(redact({'provenance':{'origin':plan['endpoint'],'project_id':plan['project']['id'],'workspace_id':plan['workspace']['id'],'selected_run_id':root['id'],'plan_id':plan['id'],'digest':plan['digest'],'imported_at':now(),'start_time':root.get('start_time'),'end_time':root.get('end_time'),'source':'LangSmith remote historical trace','executed_by_agentqa':False},'spans':spans,'feedback':feedback}))
    store.put('evidence',evidence);result={'id':uid(),'at':now(),'plan_id':plan['id'],'evidence_id':evidence.id,'trace_id':evidence.trace_id,'source':'historical','span_count':len(spans),'feedback_count':len(feedback),'target_invocations':0,'model_calls':0}
    store.put('langsmith_trace_import',result);store.audit(actor,'historical_trace_import',result);return result
