import pytest
from agentqa.models import Assertion, Evidence, TestCase, Verdict
from agentqa.evaluation import evaluate,aggregate


def case(a,requires=None):return TestCase(name='Check',objective='Check',capability='stock',category='functional',turns=['Hello'],expected_behavior='Expected',assertions=[a],evidence_requirements=requires or [])

@pytest.mark.parametrize('kind,path,expected,data',[
('equals','final','hello',{'final':'hello'}),('contains','final','ell',{'final':'hello'}),('not_contains','final','bad',{'final':'hello'}),('exists','final',None,{'final':'hello'}),('schema','raw',{'type':'object'},{'raw':{}}),('max','latency_ms',10,{'latency_ms':3}),('min','latency_ms',1,{'latency_ms':3}),('count','tool_calls',0,{'tool_calls':[]}),('unique','events',None,{'events':[{'a':1}]}),('tool_selection','tool_calls.0.name','lookup',{'tool_calls':[{'name':'lookup'}]}),('tool_arguments','tool_calls.0.arguments',{'a':1},{'tool_calls':[{'arguments':{'a':1}}]}),('required_arguments','raw',['a'],{'raw':{'a':1}}),('operation_routing','raw.operation','stock',{'raw':{'operation':'stock'}}),('error_code','error.code','403',{'error':{'code':'403'}}),('state_invariant','state_before',{'stock':12},{'state_before':{'stock':12}}),('response_format','final',{'type':'object'},{'final':'{}'}),('latency','latency_ms',100,{'latency_ms':4}),('calculation','raw.total',24,{'raw':{'total':24}}),('thread_isolation','final','No memory',{'final':'No memory'})])
def test_assertion_kinds(kind,path,expected,data):
    result=evaluate(case(Assertion(kind=kind,path=path,expected=expected)),Evidence(source='mock',**data))
    assert result[0].verdict==Verdict.PASS

@pytest.mark.parametrize('kind',['equals','schema','latency','tool_selection','authorization_denied'])
def test_missing_evidence_inconclusive(kind):
    finding=evaluate(case(Assertion(kind=kind,path='tool_calls')),Evidence(source='mock'))
    assert finding[0].verdict==Verdict.INCONCLUSIVE


def test_authorization_cannot_use_prose():
    c=case(Assertion(kind='authorization_denied',path='tool_calls',security=True))
    assert evaluate(c,Evidence(source='mock',final='Access denied'))[0].verdict==Verdict.INCONCLUSIVE
    assert evaluate(c,Evidence(source='mock',tool_calls=[{'result':{'code':'forbidden'}}]))[0].verdict==Verdict.PASS
    assert evaluate(c,Evidence(source='mock',tool_calls=[{'result':{'code':'success'}}]))[0].verdict==Verdict.FAIL


def test_missing_required_evidence_inconclusive():
    c=case(Assertion(kind='exists',path='final'),['tool_calls'])
    findings=evaluate(c,Evidence(source='mock',final='yes'))
    assert aggregate([x.verdict for x in findings])==Verdict.INCONCLUSIVE


def test_security_failure_dominates_judge_pass():assert aggregate([Verdict.FAIL,Verdict.PASS])==Verdict.FAIL


def test_multi_turn_sequence():
    c=case(Assertion(kind='equals',path='turns.1.final',expected='violet'))
    assert evaluate(c,Evidence(source='mock',turns=[{'final':'Remembered'},{'final':'violet'}]))[0].verdict==Verdict.PASS
