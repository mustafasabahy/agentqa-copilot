"""LangSmith SDK integration. Imported traces never invoke target agents."""
import json
import os
from .models import Evidence, uid, now
from .security import redact
from .adapters import normalize_tool_result


def client():
    if not os.getenv('LANGSMITH_API_KEY'): raise ValueError('LANGSMITH_API_KEY is not configured')
    from langsmith import Client
    from .remote_approval import CURRENT_REMOTE,ApprovedSession
    from .security import PolicyError
    ctx=CURRENT_REMOTE.get()
    if not ctx:raise PolicyError('Separate LangSmith approval required; no remote request dispatched')
    from urllib3.util.retry import Retry
    return Client(session=ApprovedSession(ctx),retry_config=Retry(total=0),auto_batch_tracing=False,timeout_ms=15000)


def normalize_trace(root,children):
    def val(obj,key,default=None): return obj.get(key,default) if isinstance(obj,dict) else getattr(obj,key,default)
    calls=[]; events=[]
    for span in children:
        events.append({'id':str(val(span,'id')),'parent_run_id':str(val(span,'parent_run_id')),'name':val(span,'name'),'run_type':val(span,'run_type')})
        if val(span,'run_type')=='tool':
            calls.append({'id':str(val(span,'id')),'name':val(span,'name'),'arguments':redact(val(span,'inputs',{})),'result':redact(normalize_tool_result(val(span,'outputs')))})
    outputs=val(root,'outputs') or {}; final=outputs.get('output',outputs.get('response',outputs.get('final')))
    if final is None and outputs.get('messages'):
        message=outputs['messages'][-1]
        final=message.get('content') if isinstance(message,dict) else str(message)
    return Evidence(source='historical',final=str(final) if final is not None else None,tool_calls=calls if children else None,events=events,trace_id=str(val(root,'id')),trace_url=val(root,'url') or ('https://smith.langchain.com/r/'+str(val(root,'id'))),raw=redact({'inputs':val(root,'inputs'),'outputs':outputs}),usage={k:val(root,k) for k in ('total_tokens','prompt_tokens','completion_tokens','total_cost') if val(root,k) is not None})


def projects():
    # A single explicit metadata request avoids SDK discovery and hidden pagination.
    from .remote_approval import CURRENT_REMOTE,ApprovedSession
    from .security import PolicyError
    ctx=CURRENT_REMOTE.get()
    if not ctx or ctx['plan']['action']!='projects':raise PolicyError('Exact read-only project-list approval required')
    p=ctx['plan'];key=os.getenv('LANGSMITH_API_KEY')
    if not key:raise PolicyError('LangSmith credential missing; no request dispatched')
    headers={'x-api-key':key}
    if p['workspace']['id']:headers['X-Tenant-Id']=p['workspace']['id']
    with ApprovedSession(ctx) as session:
        response=session.request('GET',p['endpoint']+p['request']['path'],params=p['request']['query'],headers=headers,timeout=p['timeout_seconds'])
        if response.status_code!=200:raise PolicyError('LangSmith project listing failed with HTTP '+str(response.status_code)+'; no retry authorized')
        raw=response.json()
    if not isinstance(raw,list) or len(raw)>100:raise PolicyError('Unexpected project-list response; no retry authorized')
    result=[{'id':str(item['id']),'name':redact(item.get('name','')),'workspace_id':str(item['tenant_id']) if item.get('tenant_id') else None} for item in raw]
    ctx['store'].put('langsmith_project_listing',{'id':uid(),'at':now(),'plan_id':p['id'],'endpoint':p['endpoint'],'source':'LangSmith remote project metadata','projects':result,'complete_catalog':len(result)<100,'target_invocations':0})
    return result


def import_runs(project,store,limit=20):
    c=client(); imported=[]
    for root in c.list_runs(project_name=project,is_root=True,limit=min(limit,100)):
        tree=c.read_run(root.id,load_child_runs=True)
        def walk(run):
            children=getattr(run,'child_runs',None) or []
            return children+[g for child in children for g in walk(child)]
        e=normalize_trace(tree,walk(tree)); store.put('evidence',e); imported.append(e.id)
    return {'imported':imported,'source':'historical','target_invocations':0}


def feedback(trace_id,key,score,comment):
    r=client().create_feedback(run_id=trace_id,key=key,score=score,comment=comment)
    return {'id':str(r.id)}


def create_dataset(name,examples):
    c=client(); dataset=c.create_dataset(dataset_name=name,description='AgentQA explicitly requested QA dataset')
    for item in examples: c.create_example(inputs=item['inputs'],outputs=item.get('outputs'),dataset_id=dataset.id)
    return {'id':str(dataset.id),'name':name}


def record_experiment(name,metadata):
    p=client().create_project(project_name=name,description='AgentQA experiment metadata',metadata=metadata)
    return {'id':str(p.id),'name':name}


def import_selected(run_ids,store):
    if not 1<=len(run_ids)<=3:raise ValueError('Select 1..3 explicit run IDs')
    c=client();imported=[]
    for id in run_ids:
        tree=c.read_run(id,load_child_runs=True)
        def walk(run):
            children=getattr(run,'child_runs',None) or []
            return children+[g for child in children for g in walk(child)]
        e=normalize_trace(tree,walk(tree));store.put('evidence',e);imported.append(e.id)
    return {'imported':imported,'source':'historical','target_invocations':0}
