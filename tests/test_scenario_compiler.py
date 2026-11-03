from agentqa.discovery import REFERENCE
import pytest
import json
from pathlib import Path
from agentqa import scenario_compiler as compiler,compiler_demo,llm
from agentqa.models import Agent,Evidence
from agentqa.discovery import source_profile
from agentqa.evaluation import evaluate

@pytest.fixture
def source(store):
    if not REFERENCE.exists(): pytest.skip("Optional private HR reference is not configured")
    a=Agent(id='source',name='HR source',adapter='source',profile=source_profile());store.put('agent',a);return a

def proposal(**changes):return {'operation_id':'get_leave_balance','objective':'Optional year','user_inputs':['Read E001 leave balance'],'arguments':{'employee_id':'E001'},**changes}

def test_compile_contract_owns_executable_fields(source):
    c=compiler.compile_proposal(source,proposal())
    assert c.capability=='get_leave_balance' and c.risk=='read_only' and not c.reviewed
    assert 'live' not in c.modes and 'multi_turn' not in c.modes
    assert all('result' not in a.path for a in c.assertions)
    assert any(f.verdict=='INCONCLUSIVE' for f in evaluate(c,Evidence(source='mock')))

@pytest.mark.parametrize('args',[{'employee_id':'E001'},{'employee_id':'E001','year':None},{'employee_id':'E001','year':2027}])
def test_nullable_optional_arguments(source,args):assert compiler.compile_proposal(source,proposal(arguments=args))

@pytest.mark.parametrize('raw',[proposal(operation_id='leave_balance_lookup'),proposal(arguments={'employee_id':'E001','year':'2027'}),proposal(arguments={}),proposal(modes=['live']),proposal(expected_json=None),proposal(risk='read_only'),proposal(assertions=[{'path':'fake'}])])
def test_bad_proposals_rejected(source,raw):
    with pytest.raises(Exception):compiler.compile_proposal(source,raw)

def test_actual_rejected_payload_never_becomes_executable(source):
    recorded=json.loads((Path(__file__).parent/'fixtures/rejected_model_payload.json').read_text())
    for c in recorded['cases']:
        with pytest.raises(Exception):compiler.compile_proposal(source,c)

def test_missing_required_is_not_optional(source):
    assert compiler.compile_proposal(source,proposal(arguments={},intent='missing_required',missing_argument='employee_id'))
    with pytest.raises(Exception):compiler.compile_proposal(source,proposal(intent='missing_required',missing_argument='year'))

@pytest.mark.parametrize('count',[3,40])
async def test_public_graph_compiler_persistence_without_provider(store,source,count,monkeypatch):
    monkeypatch.setattr(llm,'model',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No provider')))
    r=await compiler_demo.run(store,source.id,'human',count)
    assert r['provider_calls']==0 and r['compilation']['persisted']==count and r['compilation']['reviewed']==0
    assert all('live' not in c['modes'] for c in store.get('suite',r['compilation']['suite_id'])['cases'])

async def test_partial_preserves_unrelated_valid_drafts(store,source):
    async def model(*a):return {'proposals':[proposal(),{'operation_id':'invented'},proposal(objective='Null year',arguments={'employee_id':'E001','year':None},user_inputs=['Read E001 with null year'])]}
    r=await compiler.generate(store,source,'human',3,'',[],model,'controlled')
    assert r['persisted']==2 and r['generated']==3 and len(r['rejected'])==1 and r['status']=='partial'

async def test_duplicate_no_refill(store,source):
    calls=[]
    async def model(*a):calls.append(1);return {'proposals':[proposal(),proposal(),proposal()]}
    r=await compiler.generate(store,source,'human',3,'',[],model,'controlled')
    assert r['persisted']==1 and len(calls)==1

@pytest.mark.parametrize('count',[1,10])
async def test_additional_exact_counts(store,source,count):
    r=await compiler_demo.run(store,source.id,'human',count)
    assert r['compilation']['persisted']==count

@pytest.mark.parametrize('count',[0,41,-1,True,3.0,'3','Propose 1..40 cases'])
def test_intent_request_count_strict(count):
    with pytest.raises(Exception):compiler.GenerationRequest(agent_id='a',count=count)

async def test_tool_no_implicit_retry_or_count_change(store,source):
    calls=[]
    async def fixture(*a):calls.append(1);return {'proposals':[proposal()]}
    tool=llm.grounded_copilot_tools(store,source.id,'human',fixture,1)[1]
    with pytest.raises(ValueError,match='Count must match'):await tool.ainvoke({'agent_id':source.id,'count':3})
    assert not calls
    assert (await tool.ainvoke({'agent_id':source.id,'count':1}))['persisted']==1
    with pytest.raises(ValueError,match='already attempted'):await tool.ainvoke({'agent_id':source.id,'count':1})
    assert len(calls)==1

def test_independent_immutable_judge_flow(store,source):
    from agentqa import provider
    from agentqa.models import TestCase,Evidence
    case=TestCase(name='Recorded denial',objective='Respect recorded denial',category='authorization',turns=['Read E001'],expected_behavior='Respect observed denial',capability='get_leave_balance')
    store.put('golden',{'id':'golden-hr-sara-authorization','test_case':case.model_dump(mode='json'),'evidence':Evidence(source='historical',final='Denied').model_dump(mode='json')})
    generation=provider.propose_minimal(store,'human',source.id,'generation')
    judge=provider.propose_minimal(store,'human',source.id,'judge')
    assert generation['max_calls']==5 and judge['max_calls']==1
    assert generation['budget_usd']+judge['budget_usd']<=.20
    assert [o['role'] for o in judge['operations']]==['judge']
    assert generation['golden_ids']==[] and judge['golden_ids']==['golden-hr-sara-authorization']
    assert not store.all('provider_approval') and not store.all('usage')
    provider.approve(store,'human',generation['id'],generation['digest'])
    provider.stop_batch(store,'human',generation['id'],'Controlled generation failure')
    assert provider.reviewed_plan_export(store.get('provider_plan',judge['id']))==json.loads(json.dumps(judge))
    from agentqa import judge_contract
    assert judge_contract.fixture(case,Evidence(source='historical',final='Denied'))['schema_valid']
    assert provider.implementation_manifest(store,source.id,'human')['generation_system']==llm.SCENARIO_SYSTEM

async def test_partial_graph_does_not_claim_full_success(store,source):
    r=await compiler_demo.run(store,source.id,'human',3,True)
    assert not r['chat']['complete'] and 'Generation incomplete' in r['chat']['answer']
    assert r['compilation']['persisted']==2 and not store.all('provider_approval')
