from agentqa.discovery import REFERENCE
import pytest
import ast
import jsonschema
from agentqa.discovery import source_profile,annotation_schema
from agentqa.generation import generate,quality_review,validate_case
from agentqa.models import Agent,Assertion,Evidence,TestCase,Finding,JudgeFinding,Result
from agentqa.evaluation import evaluate
from agentqa.analysis import failure_attribution
from agentqa.capabilities import execution_capabilities

@pytest.fixture
def profile():
    if not REFERENCE.exists(): pytest.skip("Optional private HR reference is not configured")
    return source_profile()
@pytest.mark.parametrize('value,valid',[
 ({'employee_id':'E001'},True),({'employee_id':'E001','year':None},True),({'employee_id':'E001','year':2027},True),({'employee_id':'E001','year':'2027'},False),({'employee_id':'E001','year':1800},False),({'year':2027},False)])
def test_balance_contract_fidelity(profile,value,valid):
    cap=next(c for c in profile.capabilities if c.name=='get_leave_balance')
    assert jsonschema.Draft202012Validator(cap.schema_).is_valid(value)==valid
    year=cap.schema_['properties']['year'];assert year['default'] is None
    assert 'omitted/null uses current year' in year['description'].lower()
    assert 'x-provenance' in year

def test_union_annotation():
    assert annotation_schema(ast.parse('int | None',mode='eval').body)=={'anyOf':[{'type':'integer'},{'type':'null'}]}

def test_invalid_optional_argument_oracle_rejected(profile):
    case=TestCase(name='Bad oracle',objective='Check optional year',capability='get_leave_balance',category='missing_arguments',turns=['What is my annual leave balance? I have not specified a year.'],expected_behavior='Ask for year',assertions=[Assertion(kind='count',path='tool_calls',expected=0)],risk='read_only',modes=['static'])
    assert quality_review(case,profile)['errors']
    with pytest.raises(ValueError,match='Semantic oracle'):validate_case(case,profile)

def test_corrected_generation_distinguishes_default_null_required_and_invalid(profile):
    suite=generate(Agent(name='HR',adapter='source',profile=profile),40)
    cases=[c for c in suite.cases if c.capability=='get_leave_balance']
    assert {'optional_defaults','nullable_arguments','missing_arguments','invalid_values','argument_extraction'}<={c.category for c in cases}
    missing=next(c for c in cases if c.category=='missing_arguments')
    assert 'No employee ID' in missing.turns[0]
    assert not any('which year' in c.expected_behavior.lower() for c in cases)
    assert not any(c.reviewed for c in suite.cases)
    assert all(not quality_review(c,profile)['errors'] for c in suite.cases)

def result(verdict='INCONCLUSIVE',criterion='evidence:tool_calls',judges=None,error=None):
    return Result(test_case_id='case',verdict=verdict,findings=[Finding(criterion=criterion,verdict=verdict,explanation='test',evidence_refs=[criterion])],judge_findings=judges or [],evidence=Evidence(source='historical'),error_category=error).model_dump(mode='json')
@pytest.mark.parametrize('criterion,category',[
 ('evidence:tool_calls','missing_tool_observation'),('evidence:raw.independent_clarification_review','missing_reviewer_evidence'),('evidence:final','missing_execution_evidence')])
def test_missing_evidence_not_judge_uncertainty(criterion,category):
    actual=failure_attribution(result(criterion=criterion));assert actual['category']==category and not actual['judge_ran'] and actual['source']=='historical'

def test_actual_judge_uncertainty_only_with_judge_output():
    item=result();item['findings']=[];item['judge_findings']=[{'verdict':'INCONCLUSIVE','uncertainty':True}]
    assert failure_attribution(item)['category']=='judge_uncertainty'
@pytest.mark.parametrize('error,category',[('provider_authentication','model_execution_error'),('connection','adapter_connection_error')])
def test_error_provenance(error,category):assert failure_attribution(result(verdict='ERROR',error=error))['category']==category

def test_null_observation_is_not_missing():
    case=TestCase(name='Null',objective='Check explicit null',capability='get_leave_balance',category='nullable_arguments',turns=['Use null'],expected_behavior='Uses default',assertions=[Assertion(kind='equals',path='tool_calls.0.arguments.year',expected=None)])
    assert evaluate(case,Evidence(source='mock',tool_calls=[{'arguments':{'year':None}}]))[0].verdict=='PASS'
    assert evaluate(case,Evidence(source='mock',tool_calls=[{'arguments':{}}]))[0].verdict=='INCONCLUSIVE'

def test_source_adapter_modes_are_accurate(profile):
    a=Agent(name='HR',adapter='source',profile=profile)
    c=execution_capabilities(a)
    assert 'live' in c['underlying_agent_modes'] and 'live' not in c['adapter_supported_modes']
    assert 'live' not in profile.supported_modes and not c['mode_states'][4]['selectable']
    assert 'live' in c['blocked_modes']
