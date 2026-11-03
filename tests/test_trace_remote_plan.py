import pytest
from types import SimpleNamespace
from agentqa import remote_approval as remote,trace_remote
from agentqa.security import PolicyError
RUN='11111111-1111-4111-8111-111111111111'
PROJECT='22222222-2222-4222-8222-222222222222'
WORKSPACE='33333333-3333-4333-8333-333333333333'
@pytest.fixture
def traceplan(store):
    store.put('langsmith_project_listing',{'id':'listing','projects':[{'id':PROJECT,'workspace_id':WORKSPACE,'name':'fixture'}]})
    store.put('golden',{'id':'fixture','evidence':{'trace_id':RUN}})
    return remote.propose(store,'human','trace_read',{'project_id':PROJECT,'workspace_id':WORKSPACE,'run_id':RUN})

def test_trace_read_unapproved_no_network(store,traceplan,monkeypatch):
    monkeypatch.setattr('requests.Session.request',lambda *a,**k:pytest.fail('No remote access'))
    with pytest.raises(KeyError):
        with remote.scope(store,'human',traceplan['id'],'trace_read',traceplan['payload']):pass
    assert traceplan['max_http_requests']==3 and traceplan['requests'][1]['method']=='POST'

@pytest.mark.parametrize('wrong',['method','path','query','body'])
def test_trace_read_exact_order_rejects_mutation(store,traceplan,monkeypatch,wrong):
    remote.approve(store,'human',traceplan['id'],traceplan['digest'])
    monkeypatch.setattr('requests.Session.request',lambda *a,**k:pytest.fail('No remote access'))
    method='GET';url=traceplan['endpoint']+traceplan['requests'][0]['path'];kwargs={'params':{}}
    if wrong=='method':method='DELETE'
    if wrong=='path':url+='/other'
    if wrong=='query':kwargs['params']={'all':True}
    if wrong=='body':kwargs['json']={'private':'content'}
    with remote.scope(store,'human',traceplan['id'],'trace_read',traceplan['payload']):
        with pytest.raises(PolicyError):remote.ApprovedSession(remote.CURRENT_REMOTE.get()).request(method,url,**kwargs)
    assert store.get('remote_approval',traceplan['id'])['requests']==0

def test_trace_read_controlled_three_reads_historical_import(store,traceplan,monkeypatch):
    remote.approve(store,'human',traceplan['id'],traceplan['digest']);monkeypatch.setenv('LANGSMITH_API_KEY','fixture')
    root={'id':RUN,'trace_id':RUN,'session_id':PROJECT,'outputs':{'output':'Historical response'},'start_time':'2026-11-01T00:00:00Z','end_time':'2026-11-01T00:00:01Z'}
    tool={'id':'tool','trace_id':RUN,'session_id':PROJECT,'parent_run_id':RUN,'run_type':'tool','name':'hr_operations','inputs':{'operation':'get_leave_balance'},'outputs':{'ok':False},'start_time':'2026-11-01T00:00:00Z'}
    values=iter([root,{'runs':[root,tool]},[]]);calls=[]
    def fake(*a,**k):calls.append((a,k));return SimpleNamespace(status_code=200,json=lambda:next(values))
    monkeypatch.setattr('requests.Session.request',fake)
    with remote.scope(store,'human',traceplan['id'],'trace_read',traceplan['payload']):result=trace_remote.retrieve(store,'human',traceplan)
    assert len(calls)==3 and result['source']=='historical' and result['target_invocations']==0
    e=store.get('evidence',result['evidence_id']);assert e['trace_id']==RUN and e['raw']['provenance']['start_time']==root['start_time']
    assert not store.all('run') and not store.all('usage')

def test_trace_mismatch_stops_before_query(store,traceplan,monkeypatch):
    remote.approve(store,'human',traceplan['id'],traceplan['digest']);monkeypatch.setenv('LANGSMITH_API_KEY','fixture');calls=[]
    def fake(*a,**k):calls.append(1);return SimpleNamespace(status_code=200,json=lambda:{'id':RUN,'session_id':'wrong','trace_id':RUN})
    monkeypatch.setattr('requests.Session.request',fake)
    with remote.scope(store,'human',traceplan['id'],'trace_read',traceplan['payload']):
        with pytest.raises(PolicyError,match='mismatch'):trace_remote.retrieve(store,'human',traceplan)
    assert len(calls)==1 and not store.all('evidence')
