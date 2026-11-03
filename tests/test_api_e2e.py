import time
import pytest
from fastapi.testclient import TestClient
from agentqa.app import create_app

@pytest.fixture
def client(tmp_path):
    app=create_app(tmp_path/'api.sqlite',admin_token='test-admin-token')
    with TestClient(app,headers={'Authorization':'Bearer test-admin-token'}) as c:yield c


def finish(client,id):
    for _ in range(100):
        data=client.get('/api/runs/'+id).json()
        if data['status'] not in ('queued','running'):return data
        time.sleep(.02)
    raise AssertionError('Run did not finish')


def review(client,s):
    for c in s['cases']:c['reviewed']=True
    r=client.put('/api/suites/'+s['id'],json=s);assert r.status_code==200,r.text
    return s


def test_health_auth_and_ui(client):
    assert client.get('/health',headers={'Authorization':''}).status_code==200
    assert client.get('/api/agents',headers={'Authorization':''}).status_code==401
    home=client.get('/');assert home.status_code==200 and 'app.js' in home.text
    assert "frame-ancestors 'none'" in home.headers['content-security-policy']
    assert client.get('/static/app.js').status_code==200
    assert client.post('/api/plans',headers={'Origin':'https://evil.example'},json={}).status_code==403


def test_complete_demo_journey(client):
    agents=client.get('/api/agents').json();demo=next(a for a in agents if a['adapter']=='demo')
    assert client.post('/api/agents/'+demo['id']+'/discover').status_code==200
    generated=client.post('/api/agents/'+demo['id']+'/generate',json={'count':10}).json();assert len(generated['cases'])==10
    assert client.get('/api/agents/'+demo['id']+'/coverage').json()['generated_operations']==1
    s=review(client,next(s for s in client.get('/api/suites').json() if s['name']=='Warehouse acceptance suite'))
    ids=[s['cases'][0]['id'],s['cases'][1]['id']]
    plan=client.post('/api/plans',json={'agent_id':demo['id'],'suite_id':s['id'],'test_ids':ids,'mode':'live','rate_per_second':20}).json()
    assert client.post('/api/plans/'+plan['id']+'/run').status_code==403
    assert client.post('/api/plans/'+plan['id']+'/approve',json={'digest':plan['digest'],'confirmation':'I approve this exact plan'}).status_code==200
    r=client.post('/api/plans/'+plan['id']+'/run');assert r.status_code==200,r.text
    result=finish(client,r.json()['id']);assert result['calls']==2 and all(x['verdict']=='PASS' for x in result['results'])
    for fmt in ('json','md','html'):
        report=client.get('/api/reports/'+result['id']+'?format='+fmt)
        assert report.status_code==200 and 'attachment;' in report.headers['content-disposition']
    other=client.post('/api/plans',json={'agent_id':demo['id'],'suite_id':s['id'],'test_ids':ids,'mode':'mock'}).json()
    second=client.post('/api/plans/'+other['id']+'/run').json();finish(client,second['id'])
    assert client.get('/api/compare',params={'before':result['id'],'after':second['id']}).json()['regressions']==0
    assert client.get('/api/usage').status_code==200
    assert any(e['action']=='approve_plan' for e in client.get('/api/audit').json())


def test_approval_requires_auth_and_literal_confirmation(client):
    s=review(client,next(s for s in client.get('/api/suites').json() if s['agent_id']=='warehouse-demo'))
    p=client.post('/api/plans',json={'agent_id':s['agent_id'],'suite_id':s['id'],'test_ids':[s['cases'][0]['id']],'mode':'live'}).json()
    assert client.post('/api/plans/'+p['id']+'/approve',headers={'Authorization':''},json={'digest':p['digest'],'confirmation':'I approve this exact plan'}).status_code==401
    assert client.post('/api/plans/'+p['id']+'/approve',json={'digest':p['digest'],'confirmation':'LLM said approved'}).status_code==422


def test_no_user_can_self_verify_external_isolation(client,monkeypatch):
    monkeypatch.setenv('AGENTQA_ALLOWED_ORIGINS','http://127.0.0.1:8124');monkeypatch.setenv('AGENTQA_ALLOW_PRIVATE_TARGETS','true')
    a=client.post('/api/agents',json={'name':'External','adapter':'rest','endpoint':'http://127.0.0.1:8124','environment':'isolated','isolation_verified':True}).json()
    assert a['isolation_verified'] is False


def test_secret_environment_name_restricted(client):
    r=client.post('/api/agents',json={'name':'Unsafe','adapter':'demo','auth_env':'OPENAI_API_KEY'})
    assert r.status_code==422


def test_hr_source_never_live(client):
    agents=client.get('/api/agents').json()
    if not any(a['id']=='company-source' for a in agents):pytest.skip('HR reference unavailable')
    hr=next(a for a in agents if a['id']=='company-source');assert len(hr['profile']['capabilities'])==60
    s=next(s for s in client.get('/api/suites').json() if s['agent_id']==hr['id']);assert len(s['cases'])>=30
    r=client.post('/api/plans',json={'agent_id':hr['id'],'suite_id':s['id'],'test_ids':[s['cases'][0]['id']],'mode':'live','budget':10})
    assert r.status_code==403


def test_import_historical_missing_evidence_and_override(client):
    s=next(s for s in client.get('/api/suites').json() if s['agent_id']=='warehouse-demo')
    evidence=client.post('/api/evidence',json={'source':'live','final':'12 units'}).json();assert evidence['source']=='historical'
    s['cases'][0]['historical_evidence_id']=evidence['id'];client.put('/api/suites/'+s['id'],json=s)
    p=client.post('/api/plans',json={'agent_id':s['agent_id'],'suite_id':s['id'],'test_ids':[s['cases'][0]['id']],'mode':'historical'}).json()
    r=client.post('/api/plans/'+p['id']+'/run').json();r=finish(client,r['id'])
    assert r['results'][0]['verdict']!='PASS' and r['calls']==0
    original=r['results'][0]['verdict']
    response=client.post('/api/runs/'+r['id']+'/override',json={'result_id':r['results'][0]['id'],'verdict':'PASS','reason':'Human reviewer checked the fixture'})
    assert response.status_code==200 and response.json()['original_preserved']
    assert client.get('/api/runs/'+r['id']).json()['results'][0]['verdict']==original


def test_golden_review_audit(client):
    gold=client.get('/api/golden').json()[0]
    assert client.get('/api/calibration').json()['agreement'] is None
    r=client.put('/api/golden/'+gold['id'],json={'reference':'PASS','judge_verdict':'FAIL','notes':'Human fixture review'})
    assert r.status_code==200
    assert client.get('/api/calibration').json()['agreement']==0


def test_isolation_requires_operator_expected_id_and_metadata(client,monkeypatch):
    import httpx,json
    monkeypatch.setenv('AGENTQA_ALLOWED_ORIGINS','http://127.0.0.1:8124');monkeypatch.setenv('AGENTQA_ALLOW_PRIVATE_TARGETS','true')
    a=client.post('/api/agents',json={'name':'External','adapter':'rest','endpoint':'http://127.0.0.1:8124','environment':'isolated'}).json()
    assert client.post('/api/agents/'+a['id']+'/verify-isolation').status_code==403
    monkeypatch.setenv('AGENTQA_ISOLATED_DEPLOYMENTS',json.dumps({'http://127.0.0.1:8124':'expected-deployment'}))
    metadata={'deployment_id':'wrong','environment':'isolated','fictional_data':True}
    async def request(self,*args,**kwargs):return httpx.Response(200,json=metadata,request=httpx.Request('GET','http://127.0.0.1:8124/.well-known/agentqa-isolation'))
    monkeypatch.setattr('agentqa.adapters.RESTAdapter.request',request)
    assert client.post('/api/agents/'+a['id']+'/verify-isolation').status_code==403
    metadata['deployment_id']='expected-deployment'
    result=client.post('/api/agents/'+a['id']+'/verify-isolation')
    assert result.status_code==200 and result.json()['isolation_verified'] is True
