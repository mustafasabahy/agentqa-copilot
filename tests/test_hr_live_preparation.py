from agentqa.discovery import REFERENCE
import pytest
from agentqa.hr_live_preparation import prepare,verify
pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="Optional private HR reference is not configured")
from agentqa.models import Agent
from agentqa.provider import digest
from agentqa.security import PolicyError

def test_preparation_is_immutable_non_executable_no_execution(store):
    store.put('agent',Agent(id='company-source',name='Source',adapter='source'))
    p=prepare(store,'human')
    assert p['status']=='BLOCKED' and not p['executable'] and p['limits']['authorized_calls_now']==0
    assert len(p['tests'])==3 and sum(t['max_target_invocations'] for t in p['tests'])==4
    assert p['limits']['proposed_model_calls_total']==12 and p['limits']['proposed_total_target_http_requests']==10
    assert verify(p)==p and not store.all('plan') and not store.all('approval') and not store.all('run') and not store.all('usage')
    assert p['digest']==digest({k:v for k,v in p.items() if k!='digest'})
    p['limits']['proposed_inference_target_calls']=100
    with pytest.raises(PolicyError,match='digest'):verify(p)

def test_expired_preparation_cannot_be_reviewed_as_current(store):
    store.put('agent',Agent(id='company-source',name='Source',adapter='source'));p=prepare(store,'human');p['expires_at']='2000-01-01T00:00:00+00:00';p['digest']=digest({k:v for k,v in p.items() if k!='digest'})
    with pytest.raises(PolicyError,match='expired'):verify(p)

def test_backend_export_preserves_digest_and_approval_stays_blocked(tmp_path):
    from fastapi.testclient import TestClient
    from agentqa.app import create_app
    app=create_app(tmp_path/'preparation.sqlite',admin_token='fixture-admin',seed_data=False)
    app.state.store.put('agent',Agent(id='company-source',name='Source',adapter='source'))
    with TestClient(app,headers={'Authorization':'Bearer fixture-admin'}) as client:
        p=client.post('/api/hr-live/preparations').json()
        response=client.get('/api/hr-live/preparations/'+p['id']+'/export')
        assert response.status_code==200 and verify(response.json())['digest']==p['digest']
        assert b'1.0' in response.content
        denied=client.post('/api/hr-live/preparations/'+p['id']+'/approve')
        assert denied.status_code==403 and 'BLOCKED' in denied.json()['detail']
    assert not app.state.store.all('approval') and not app.state.store.all('run')
