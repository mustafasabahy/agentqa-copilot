from agentqa.adapters import pick
from agentqa.models import Assertion,Evidence,TestCase
from agentqa.evaluation import evaluate

def test_metadata_calls_do_not_hide_correct_routing():
    calls=[{'name':'describe_operation','arguments':{'operation':'get_leave_balance'}},{'name':'get_employee','arguments':{'name':'Sara'}},{'name':'hr_operations','arguments':{'operation':'get_leave_balance','arguments':{'employee_id':'E002','year':2027}},'result':{'code':'forbidden'}}]
    # Operation selector finds the BUSINESS call; metadata operation names alone must not match.
    assert pick({'tool_calls':calls},'tool_calls[name=hr_operations].arguments.arguments.year')==2027
    c=TestCase(name='Routing',objective='Routing',capability='get_leave_balance',category='functional',turns=['2027'],expected_behavior='Denied after correct routing',assertions=[Assertion(kind='tool_selection',path='tool_calls',expected='hr_operations'),Assertion(kind='operation_routing',path='tool_calls',expected='get_leave_balance')])
    assert all(f.verdict=='PASS' for f in evaluate(c,Evidence(source='mock',tool_calls=calls)))
