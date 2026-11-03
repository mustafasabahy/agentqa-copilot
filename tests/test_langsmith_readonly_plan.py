import pytest
from agentqa import remote_approval as remote
from agentqa.security import PolicyError

@pytest.mark.parametrize('method,path,query,body',[
 ('POST','/sessions',None,None),('GET','/runs',None,None),('GET','/sessions',{'limit':1000},None),('GET','/sessions',None,{'trace':'private'})])
def test_exact_project_plan_rejects_scope_before_network(store,monkeypatch,method,path,query,body):
    p=remote.propose(store,'human','projects',{});remote.approve(store,'human',p['id'],p['digest'])
    monkeypatch.setattr('requests.Session.request',lambda *a,**k:pytest.fail('No remote request allowed'))
    with remote.scope(store,'human',p['id'],'projects',{}):
        with pytest.raises(PolicyError):remote.ApprovedSession(remote.CURRENT_REMOTE.get()).request(method,p['endpoint']+path,params=query or p['request']['query'],json=body)
    assert store.get('remote_approval',p['id'])['requests']==0

def test_unapproved_project_list_never_contacts_remote(store,monkeypatch):
    from agentqa import langsmith_integration as smith
    monkeypatch.setattr('requests.Session.request',lambda *a,**k:pytest.fail('No remote request allowed'))
    with pytest.raises(PolicyError):smith.projects()
    p=remote.propose(store,'human','projects',{})
    assert p['max_http_requests']==1 and p['retries']==0 and p['request']['method']=='GET'
    assert not store.all('remote_approval')

def test_approved_project_list_is_one_mocked_metadata_request(store,monkeypatch):
    from agentqa import langsmith_integration as smith
    from types import SimpleNamespace
    monkeypatch.setenv('LANGSMITH_API_KEY','local-fixture-key')
    calls=[]
    def fake(*args,**kwargs):
        calls.append((args,kwargs));return SimpleNamespace(status_code=200,json=lambda:[{'id':'project-fixture','name':'Fixture','tenant_id':'workspace-fixture','inputs':{'private':'discard'}}])
    monkeypatch.setattr('requests.Session.request',fake)
    p=remote.propose(store,'human','projects',{});remote.approve(store,'human',p['id'],p['digest'])
    with remote.scope(store,'human',p['id'],'projects',{}):
        assert smith.projects()==[{'id':'project-fixture','name':'Fixture','workspace_id':'workspace-fixture'}]
        with pytest.raises(PolicyError,match='ceiling'):smith.projects()
    assert len(calls)==1 and calls[0][1]['allow_redirects'] is False
    assert 'private' not in str(store.all('langsmith_project_listing'))
