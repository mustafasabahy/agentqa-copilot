"""Human-reviewed, bounded LangSmith remote requests; never target execution."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
import json
import hashlib
from pathlib import Path
import os
from urllib.parse import urlsplit
import requests
from .models import uid,now
from .provider import digest
from .security import PolicyError
CURRENT_REMOTE=ContextVar('remote_approval',default=None)
def implementation_hashes():
    return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('remote_approval.py','langsmith_integration.py','trace_remote.py')}

def propose(store,actor,action,payload):
    snapshots={}
    if action=='dataset':snapshots['suite']=digest(store.get('suite',payload['suite_id']))
    if action=='experiment':snapshots['run']=digest(store.get('run',payload['run_id']))
    p={'snapshots':snapshots,'endpoint':os.getenv('LANGSMITH_ENDPOINT','https://api.smith.langchain.com'),'id':uid(),'owner':actor,'action':action,'payload':payload,'max_http_requests':12,'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),'disclosure':'LangSmith remote requests only. Export actions send selected data. No target execution. No OpenAI approval.'}
    if action=='projects':
        if payload:raise PolicyError('Project listing accepts no arbitrary payload')
        endpoint=p['endpoint'].rstrip('/');parsed=urlsplit(endpoint)
        if parsed.scheme!='https' or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:raise PolicyError('Project-list endpoint must be an HTTPS origin')
        p.update({'implementation_hashes':implementation_hashes(),'endpoint':endpoint,'max_http_requests':1,'workspace':{'id':os.getenv('LANGSMITH_WORKSPACE_ID') or None,'selection':'Explicit configured workspace' if os.getenv('LANGSMITH_WORKSPACE_ID') else 'Credential default workspace; ID unverified before approved listing'},'request':{'method':'GET','path':'/sessions','query':{'limit':100,'offset':0,'include_stats':'false'},'body':None,'header_names':['x-api-key']+(['X-Tenant-Id'] if os.getenv('LANGSMITH_WORKSPACE_ID') else [])},'read_scope':['First page, up to 100 accessible project IDs, names and workspace IDs','Project metadata only; no trace inputs, outputs, tool data or evaluations'],'excluded':['Trace search/import requires a new exact plan with reviewed project/run IDs','No writes, ingestion, feedback, dataset creation or tracing','No OpenAI or HR target requests'],'cost':{'openai_usd':0,'target_usd':0,'application_spend_budget_usd':0,'langsmith_account_charges':'Not verified; existing account terms apply. No paid ingestion or model operation authorized.'},'credentials':{'configured':bool(os.getenv('LANGSMITH_API_KEY')),'remote_validity':'UNVERIFIED until this approved request succeeds','delivery':'Existing server-side credential in x-api-key header to approved HTTPS origin; never returned to UI'},'retries':0,'redirects':False,'timeout_seconds':15,'disclosure':'One read-only GET /sessions to LangSmith with the existing server-side credential and exact pagination query. No request body or local trace/test data uploaded. Workspace is explicit if configured, otherwise selected by that credential. Return project metadata only. No implicit pagination, retries, redirects, trace access, target execution or OpenAI calls.'})
    if action=='trace_read':
        from .trace_remote import configure
        p=configure(store,p,payload);p['implementation_hashes']=implementation_hashes()
    p['digest']=digest(p);store.put('remote_plan',p);store.audit(actor,'remote_plan_proposed',{'id':p['id'],'digest':p['digest'],'action':action});return p
def approve(store,actor,id,expected):
    p=store.get('remote_plan',id)
    if p['owner']!=actor or expected!=p['digest'] or digest({k:v for k,v in p.items() if k!='digest'})!=p['digest'] or datetime.fromisoformat(p['expires_at'])<datetime.now(timezone.utc):raise PolicyError('Remote approval scope mismatch')
    try:store.get('remote_approval',id)
    except KeyError:pass
    else:raise PolicyError('Remote approval already exists')
    a={'id':id,'actor':actor,'digest':expected,'at':now(),'requests':0};store.put('remote_approval',a);store.audit(actor,'approve_remote_plan',a);return a
@contextmanager
def scope(store,actor,id,action,payload):
    if not id:raise PolicyError('Separate exact LangSmith approval required; no remote request dispatched')
    p=store.get('remote_plan',id);a=store.get('remote_approval',id)
    if a['actor']!=actor or p['owner']!=actor or a['digest']!=p['digest'] or digest({k:v for k,v in p.items() if k!='digest'})!=p['digest'] or p['action']!=action or p['payload']!=payload or datetime.fromisoformat(p['expires_at'])<datetime.now(timezone.utc):raise PolicyError('Remote approval does not match request/owner/expiry')
    if action in ('projects','trace_read') and p.get('implementation_hashes')!=implementation_hashes():raise PolicyError('Read-only implementation changed; fresh approval required')
    for kind,hash_value in p.get('snapshots',{}).items():
        if digest(store.get(kind,payload[kind+'_id']))!=hash_value:raise PolicyError('Exported data changed since remote plan review')
    token=CURRENT_REMOTE.set({'store':store,'plan':p,'actor':actor})
    try:yield
    finally:CURRENT_REMOTE.reset(token)
class ApprovedSession(requests.Session):
    def __init__(self,ctx):super().__init__();self.ctx=ctx
    def request(self,method,url,*args,**kwargs):
        ctx=self.ctx;store=ctx['store'];p=ctx['plan']
        if urlsplit(url).netloc!=urlsplit(p['endpoint']).netloc or urlsplit(url).scheme!='https' or datetime.fromisoformat(p['expires_at'])<datetime.now(timezone.utc):raise PolicyError('Remote destination or expiry outside approved scope')
        if p['action']=='projects':
            r=p.get('request')
            if not r:raise PolicyError('Legacy broad project plan must be replaced with fresh exact approval')
            if method.upper()!=r['method'] or url!=p['endpoint']+r['path'] or kwargs.get('params')!=r['query'] or any(kwargs.get(k) is not None for k in ('data','json','files')):raise PolicyError('Remote method, path, query or body outside exact read-only plan')
            if (os.getenv('LANGSMITH_WORKSPACE_ID') or None)!=p['workspace']['id']:raise PolicyError('Configured workspace changed; fresh approval required')
        with store.lock,store.connect() as c:
            c.execute('BEGIN IMMEDIATE');row=c.execute("SELECT body FROM documents WHERE kind='remote_approval' AND id=?",(p['id'],)).fetchone();a=json.loads(row['body'])
            if a['requests']>=p['max_http_requests']:raise PolicyError('Remote HTTP request ceiling exhausted')
            if p['action']=='trace_read':
                r=p['requests'][a['requests']]
                if method.upper()!=r['method'] or url!=p['endpoint']+r['path'] or kwargs.get('params',{})!=r['query'] or kwargs.get('json')!=r['body'] or any(kwargs.get(k) is not None for k in ('data','files')):raise PolicyError('Trace-read request outside exact ordered plan')
            a['requests']+=1;c.execute("UPDATE documents SET body=? WHERE kind='remote_approval' AND id=?",(json.dumps(a),p['id']))
        store.audit(ctx['actor'],'langsmith_request_authorized',{'plan_id':p['id'],'method':method,'path':urlsplit(url).path,'request_number':a['requests']})
        kwargs['allow_redirects']=False
        return super().request(method,url,*args,**kwargs)
