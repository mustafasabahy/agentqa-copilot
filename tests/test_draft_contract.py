from agentqa.discovery import REFERENCE
import pytest
import json
from types import SimpleNamespace
from pydantic import ValidationError
from agentqa import draft_workflow as drafts,llm,offline_demo,judge_contract
from agentqa.models import Evidence

@pytest.mark.parametrize('count',[1,3,10,40])
async def test_exact_count_and_bounded_persistence(setup,count):
    store,a,_=setup;inv=f'invoke-{count}'
    job=drafts.start(store,a,'human',inv,count)
    assert job['requested_count']==count and max(job['batch_sizes'])<=20
    cases=offline_demo.fixture_cases(a,count)
    for index,offset in enumerate(range(0,count,20)):
        job=drafts.submit(store,a,'human',inv,{'request_id':job['id'],'batch_index':index,'cases':cases[offset:offset+20]})
    assert job['status']=='complete' and job['saved_count']==count and job['remaining_count']==0
    assert len(store.get('suite',job['suite_id'])['cases'])==count
    assert all(not c['reviewed'] for c in store.get('suite',job['suite_id'])['cases'])
    assert not store.all('usage')

@pytest.mark.parametrize('bad',[0,-1,41,1000,'40','Propose 1..40 cases',3.0,True,None,{},[]])
def test_strict_count_no_coercion(bad):
    with pytest.raises(ValidationError):drafts.DraftRequest(count=bad)

@pytest.mark.parametrize('text,expected',[
 ('Then propose 40 meaningful test cases.',40),('Generate 3 structured test cases.',3),
 ('Propose 1..40 cases',None),('Generate forty cases',None),('Year 2027; analyze tools',None),
 ('Generate 3 cases, then generate 10 cases',None)])
def test_only_explicit_unambiguous_counts(text,expected):assert drafts.explicit_requested_count(text)==expected

async def test_real_tool_schema_exposes_count_and_case_fields(setup):
    store,a,_=setup;tools={t.name:t for t in llm.copilot_tools(store,a.id,'human')}
    count=tools['propose_tests'].args_schema.model_json_schema()['properties']['count']
    assert count['type']=='integer' and count['minimum']==1 and count['maximum']==40
    schema=tools['save_proposed_tests'].args_schema.model_json_schema()
    assert schema['properties']['cases']['maxItems']==20 and '$ref' in schema['properties']['cases']['items']
    assert 'DraftCase' in schema['$defs'] and schema['$defs']['DraftCase']['additionalProperties'] is False
    with pytest.raises(ValidationError):tools['propose_tests'].invoke({'count':'Propose 1..40 cases'})
    assert not store.all('suite') or len(store.all('suite'))==1 # existing setup suite only

async def test_requested_count_not_model_guess(setup):
    store,a,_=setup;tool=next(t for t in llm.copilot_tools(store,a.id,'human',expected_count=40) if t.name=='propose_tests')
    with pytest.raises(ValueError,match='count=40'):tool.invoke({'count':3})

async def test_partial_no_retry_and_persisted_valid_drafts(setup):
    store,a,_=setup;job=drafts.start(store,a,'human','partial',3);cases=offline_demo.fixture_cases(a,2)
    result=drafts.submit(store,a,'human','partial',{'request_id':job['id'],'batch_index':0,'cases':cases})
    assert result['status']=='partial' and result['saved_count']==2 and result['remaining_count']==1
    with pytest.raises(ValueError,match='already submitted'):drafts.submit(store,a,'human','partial',{'request_id':job['id'],'batch_index':0,'cases':cases})
    assert len(store.get('suite',job['suite_id'])['cases'])==2

async def test_duplicate_is_reported_not_silently_refilled(setup):
    store,a,_=setup;job=drafts.start(store,a,'human','duplicate',3);cases=offline_demo.fixture_cases(a,3);cases[1]=cases[0]
    result=drafts.submit(store,a,'human','duplicate',{'request_id':job['id'],'batch_index':0,'cases':cases})
    assert result['status']=='partial' and result['saved_count']==2 and any('Duplicate' in i['error'] for i in result['issues'])

@pytest.mark.parametrize('change',[
 {'capability':'made_up'}, {'modes':['live']}, {'risk':'unknown','assertions':[{'kind':'count','path':'tool_calls','expected_json':'"three"'}]},
 {'assertions':[{'kind':'tool_selection','path':'tool_calls','expected_json':'"invented_tool"'}]},
 {'assertions':[{'kind':'tool_arguments','path':'tool_calls.0.arguments','expected_json':'{"sku":123}'}]}])
async def test_unsupported_expected_behavior_and_arguments(setup,change):
    store,a,_=setup;wire=offline_demo.fixture_cases(a,1)[0].model_dump(mode='json');wire.update(change)
    if change.get('modes')==['live']:
        from agentqa.models import Mode
        a.profile.supported_modes=[Mode.STATIC,Mode.MOCK]
    job=drafts.start(store,a,'human','bad',1)
    result=drafts.submit(store,a,'human','bad',{'request_id':job['id'],'batch_index':0,'cases':[wire]})
    assert result['status']=='failed' and result['saved_count']==0 and result['issues']

@pytest.mark.parametrize('count',[1,3,10,40])
async def test_offline_graph_full_workflow_no_provider(setup,monkeypatch,count):
    store,a,_=setup
    monkeypatch.setattr(llm,'model',lambda *args,**kw:(_ for _ in ()).throw(AssertionError('Provider construction forbidden')))
    result=await offline_demo.run(store,a.id,'human',count)
    assert result['provider_calls']==0 and result['result']['complete']
    assert result['result']['draft_requests'][0]['saved_count']==count
    assert store.all('copilot_tool_events') and 'controlled' in result['result']['mode']

@pytest.mark.parametrize('scenario',['malformed','partial','duplicate','unsupported'])
async def test_failure_never_fabricates_success_or_retries(setup,monkeypatch,scenario):
    store,a,_=setup
    monkeypatch.setattr(llm,'model',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('No paid retry')))
    result=await offline_demo.run(store,a.id,'human',3,scenario)
    assert not result['result']['complete'] and 'Fixture model claims' not in result['result']['answer']
    assert result['provider_calls']==0 and not [u for u in store.all('usage') if u['category']=='provider_dispatch']
    assert len(store.all('chat')[0]['messages'])==2

async def test_direct_ai_generator_partial_raises_with_persisted_counts(setup):
    store,a,_=setup
    class Fake:
        calls=0
        def with_structured_output(self,*args,**kwargs):return self
        async def ainvoke(self,messages):
            self.calls+=1
            return {'parsed':drafts.GeneratedDraftBatch(cases=offline_demo.fixture_cases(a,2)),'raw':SimpleNamespace(usage_metadata={})}
    fake=Fake()
    with pytest.raises(drafts.DraftIncomplete) as error:await llm.ai_generate(a,3,'',1,store,'human',fake)
    assert fake.calls==1 and error.value.summary['saved_count']==2 and error.value.summary['requested_count']==3

async def test_batch_replay_cannot_duplicate_writes(setup):
    store,a,_=setup;job=drafts.start(store,a,'human','once',1);payload={'request_id':job['id'],'batch_index':0,'cases':offline_demo.fixture_cases(a,1)}
    drafts.submit(store,a,'human','once',payload)
    with pytest.raises(ValueError,match='already submitted'):drafts.submit(store,a,'human','once',payload)
    assert len(store.get('suite',job['suite_id'])['cases'])==1

async def test_independent_judge_schema_and_grounding(setup):
    _,_,suite=setup;case=suite.cases[0];evidence=Evidence(source='mock')
    checked=judge_contract.fixture(case,evidence)
    assert checked['schema_valid'] and checked['model_calls']==0 and checked['dimensions']==10
    assert all(f['verdict']=='INCONCLUSIVE' for f in checked['findings'])
    data={'findings':checked['findings']}
    judge_contract.parse_output(json.dumps(data),require_dimensions=True)
    data['findings'][0]['score']='1'
    with pytest.raises(ValidationError):judge_contract.parse_output(data)

@pytest.mark.parametrize('bad',[{'findings':[]},{'findings':[{'criterion':'fake'}]},'not json'])
def test_judge_malformed_outputs_rejected(bad):
    with pytest.raises((ValidationError,ValueError)):judge_contract.parse_output(bad,require_dimensions=True)

@pytest.mark.parametrize('count',[1,3,10,40])
async def test_direct_generator_exact_counts_share_the_same_contract(setup,count):
    store,a,_=setup;all_cases=offline_demo.fixture_cases(a,count)
    class Fake:
        calls=0
        def with_structured_output(self,*args,**kwargs):return self
        async def ainvoke(self,messages):
            payload=json.loads(messages[-1]['content']);index=self.calls;self.calls+=1
            segment=all_cases[index*20:(index+1)*20]
            assert payload['count']==len(segment)
            return {'parsed':drafts.GeneratedDraftBatch(cases=segment),'raw':SimpleNamespace(usage_metadata={})}
    fake=Fake();suite=await llm.ai_generate(a,count,'',1,store,'human',fake)
    assert len(suite.cases)==count and fake.calls==(count+19)//20

async def test_direct_generation_duplicate_does_not_trigger_paid_refill(setup):
    store,a,_=setup;cases=offline_demo.fixture_cases(a,3);cases[1]=cases[0]
    class Fake:
        calls=0
        def with_structured_output(self,*a,**kw):return self
        async def ainvoke(self,messages):
            self.calls+=1
            return {'parsed':drafts.GeneratedDraftBatch(cases=cases),'raw':SimpleNamespace(usage_metadata={})}
    fake=Fake()
    with pytest.raises(drafts.DraftIncomplete) as err:await llm.ai_generate(a,3,'',1,store,'human',fake)
    assert fake.calls==1 and err.value.summary['saved_count']==2


def test_provider_schemas_have_typed_properties_offline():
    from openai.lib._pydantic import to_strict_json_schema
    from agentqa.models import JudgeOutput
    from jsonschema import Draft202012Validator
    for model in (drafts.GeneratedDraftBatch,JudgeOutput,drafts.DraftRequest,drafts.SubmitBatch):
        schema=to_strict_json_schema(model);Draft202012Validator.check_schema(schema)
        def check(node):
            if isinstance(node,dict):
                for prop in node.get('properties',{}).values():assert any(k in prop for k in ('type','$ref','anyOf','oneOf','allOf'))
                for value in node.values():check(value)
            elif isinstance(node,list):
                for value in node:check(value)
        check(schema)

async def test_minimal_plan_binds_schema_and_needs_fresh_approval(store):
    if not REFERENCE.exists(): pytest.skip("Optional private HR reference is not configured")
    from agentqa.discovery import source_profile
    from agentqa.models import Agent,TestCase
    from agentqa import provider
    from agentqa.security import PolicyError
    a=Agent(id='source-test',name='Source fixture',adapter='source',profile=source_profile());store.put('agent',a)
    case=TestCase(name='Recorded example fixture',objective='Review recorded denial',capability='get_leave_balance',category='authorization',turns=['Read E002 balance'],expected_behavior='Backend denies access')
    store.put('golden',{'id':'golden-hr-sara-authorization','test_case':case.model_dump(mode='json'),'evidence':Evidence(source='historical').model_dump(mode='json')})
    plan=provider.propose_minimal(store,'human',a.id)
    assert plan['budget_usd']==.20 and plan['max_calls']==6 and plan['maximum_estimated_cost_usd']<=.20
    assert plan['implementation_digest']==provider.digest(provider.implementation_manifest(store,a.id,'human'))
    assert plan['operations'][0]['max_calls']==4 and len(plan['golden_ids'])==1 and not plan['trace']
    assert not store.all('provider_approval')
    with pytest.raises(PolicyError,match='not been explicitly approved'):
        with provider.scope(store,'human',plan['id'],'copilot',{'agent':store.get('agent',a.id),'message':plan['prompt'],'session_id':None}):pass


def test_offline_api_exercises_persistence_and_cannot_enable_provider(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from agentqa.app import create_app
    monkeypatch.setattr(llm,'model',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('No provider calls allowed')))
    app=create_app(tmp_path/'offline-api.sqlite',admin_token='fixture-admin')
    with TestClient(app,headers={'Authorization':'Bearer fixture-admin'}) as client:
        assert client.post('/api/offline/drafts',json={'agent_id':'warehouse-demo','count':'3'}).status_code==422
        result=client.post('/api/offline/drafts',json={'agent_id':'warehouse-demo','count':3}).json()
        assert result['provider_calls']==0 and result['result']['complete']
        job=result['result']['draft_requests'][0]
        suite=next(s for s in client.get('/api/suites').json() if s['id']==job['suite_id'])
        assert len(suite['cases'])==3 and 'Controlled offline' in suite['generation']
        bad=client.post('/api/offline/drafts',json={'agent_id':'warehouse-demo','count':3,'scenario':'malformed'}).json()
        assert not bad['result']['complete']
        assert client.post('/api/agents/warehouse-demo/generate',json={'count':3,'ai':True,'budget':.2}).status_code==403
        assert client.post('/api/chat',json={'agent_id':'warehouse-demo','message':'hello','model_override':{}}).status_code==422
        assert not [u for u in app.state.store.all('usage') if u['category']=='provider_dispatch']

async def test_offline_reuses_known_recorded_prefix_without_inventing_missing_output(setup):
    store,a,_=setup
    store.put('chat',{'id':'recorded-session','mode':'model-backed','messages':[]})
    store.put('copilot_evidence',{'id':'recorded-evidence','session_id':'recorded-session','events':[{'type':'tool_call','name':'agent_profile','arguments':{},'call_id':'known-profile'},{'type':'tool_call','name':'coverage_gaps','arguments':{},'call_id':'known-coverage'}]})
    data=await offline_demo.run(store,a.id,'human',40)
    assert data['result']['complete'] and data['recorded_prefix_source']=='recorded-evidence'
    assert data['recorded_tool_calls_replayed']==['agent_profile','coverage_gaps'] and data['provider_calls']==0
    assert 'missing historical arguments' in data['source']
