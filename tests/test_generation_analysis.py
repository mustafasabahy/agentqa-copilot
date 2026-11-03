from agentqa.discovery import REFERENCE
import pytest
from agentqa.models import Agent, Assertion, Mode, Risk, Suite, TestCase
from agentqa.discovery import source_profile
from agentqa.generation import generate,validate_case,deduplicate,coverage,prioritize
from agentqa.analysis import calibration,clusters,compare,report,markdown_report,html_report
from agentqa.policy import Policy
from agentqa.runner import Runner

@pytest.mark.parametrize('count',[10,20,30,50,100,250])
def test_hr_generation_sizes(count):
    if not REFERENCE.exists(): pytest.skip("Optional private HR reference is not configured")
    a=Agent(name='HR',adapter='source',profile=source_profile());s=generate(a,count)
    assert len(s.cases)==count
    assert len({c.id for c in s.cases})==count
    assert 'deterministic' in s.generation
    for c in s.cases:validate_case(c,a.profile)

async def test_semantic_validation_and_risk(setup):
    store,a,s=setup;c=s.cases[0].model_copy(deep=True);c.capability='invented'
    with pytest.raises(ValueError,match='Unknown'):validate_case(c,a.profile)
    c=s.cases[0].model_copy(deep=True);c.assertions=[Assertion(kind='tool_selection',expected='invented')]
    with pytest.raises(ValueError,match='unknown tool'):validate_case(c,a.profile)
    c=s.cases[0].model_copy(deep=True);a.profile.capabilities[0].risk=Risk.MUTATE
    with pytest.raises(ValueError,match='risk'):validate_case(c,a.profile)

async def test_deduplication(setup):
    _,a,s=setup;c=s.cases[0]
    assert len(deduplicate([c,c.model_copy(update={'id':'different'})]))==1

async def test_generated_reviewed_live_coverage_separate(setup):
    store,a,s=setup;c=coverage(a,store.all('suite'),[])
    assert c['inventory'][0]['generated']==5 and c['inventory'][0]['reviewed']==5
    assert c['inventory'][0]['executed']==0
    p=Policy(store).propose(a,s,[s.cases[0].id],Mode.MOCK);runner=Runner(store);r=runner.start(p.id,'human');await runner.tasks[r.id]
    c=coverage(a,store.all('suite'),store.all('run'));assert c['inventory'][0]['executed']==0 and c['inventory'][0]['offline_evaluated']==1

async def test_cost_aware_selection(setup):
    store,a,s=setup
    for c in s.cases:c.estimated_cost=.25
    result=prioritize(s,.25,[])
    assert result['estimated_cost']<=.25 and len(result['selection'])==1 and result['selection'][0]['reason']

async def test_reports_comparisons_and_escaping(setup):
    store,a,s=setup;s.cases[0].turns=['<script>alert(1)</script>'];store.put('suite',s)
    runner=Runner(store);policy=Policy(store);ids=[]
    for _ in range(2):
        p=policy.propose(a,s,[s.cases[0].id],Mode.MOCK);r=runner.start(p.id,'human');await runner.tasks[r.id];ids.append(r.id)
    data=report(store,ids[0]);assert data['run']['mode']=='mock' and data['limitations']
    assert '<script>' not in html_report(data)
    assert '&lt;script&gt;' in html_report(data)
    assert 'Reproduction turns' in markdown_report(data)
    comparison=compare(store.get('run',ids[0]),store.get('run',ids[1]));assert comparison['regressions']==0

async def test_failure_clustering_honest(setup):
    store,a,s=setup;s.cases[0].mock_evidence={'final':'wrong','tool_calls':[]};store.put('suite',s)
    p=Policy(store).propose(a,s,[s.cases[0].id],Mode.MOCK);runner=Runner(store);r=runner.start(p.id,'human');await runner.tasks[r.id]
    groups=clusters(store.get('run',r.id));assert groups and all('not proven' in g['confidence'] for g in groups)


def test_calibration_requires_human_labels(store):
    store.put('golden',{'id':'g','reference':'PASS','judge_verdict':'FAIL','reviewed_by':None})
    assert calibration(store)['agreement'] is None
    store.put('golden',{'id':'g','reference':'PASS','judge_verdict':'FAIL','reviewed_by':'human'})
    assert calibration(store)['agreement']==0 and len(calibration(store)['disagreements'])==1
