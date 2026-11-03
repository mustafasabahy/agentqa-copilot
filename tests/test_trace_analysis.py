from agentqa.discovery import REFERENCE
import pytest
import json
from agentqa.models import Agent,Evidence
from agentqa.discovery import source_profile
from agentqa.trace_analysis import analyze
from agentqa.adapters import normalize_tool_result

def test_toolmessage_json_normalization_preserves_non_tool_content():
    assert normalize_tool_result({'output':{'type':'tool','content':'{"ok":false,"code":"forbidden"}'}})=={'ok':False,'code':'forbidden'}
    other={'type':'arbitrary','content':'{"count":3}'}
    assert normalize_tool_result(other)==other
    assert normalize_tool_result({'type':'tool','content':'invalid JSON'})=='invalid JSON'

def test_local_analysis_redacts_and_preserves_source(store):
    if not REFERENCE.exists(): pytest.skip("Optional private HR reference is not configured")
    store.put('agent',Agent(id='company-source',name='fixture',adapter='source',profile=source_profile()))
    root={'id':'root','trace_id':'root','parent_run_id':None,'session_id':'project','run_type':'chain','start_time':'2026-11-02T00:00:00','end_time':'2026-11-02T00:00:01','inputs':{'messages':[{'role':'user','content':'Private Person: balance for E002 year 2027'}]}}
    tool={'id':'tool','parent_run_id':'root','run_type':'tool','start_time':root['start_time'],'end_time':root['end_time']}
    e=Evidence(id='imported',source='historical',trace_id='root',final='I cannot access the requested private balance',tool_calls=[{'id':'tool','name':'hr_operations','arguments':{'operation':'get_leave_balance','arguments':{'employee_id':'E002','year':2027}},'result':{'type':'tool','tool_call_id':'call_tool','content':json.dumps({'ok':False,'code':'forbidden','error':'Private Person E002','data':{'remaining_days':123.5}})}}],raw={'provenance':{'source':'LangSmith remote historical trace','executed_by_agentqa':False},'spans':[root,tool],'feedback':[]})
    store.put('evidence',e);before=store.get('evidence',e.id)
    result=analyze(store,e.id);serialized=json.dumps(result)
    assert all(private not in serialized for private in ['Private Person','E002','123.5'])
    assert result['tool_calls'][0]['result']['code']=='forbidden'
    assert result['tool_calls'][0]['arguments']['employee_id']=='EMPLOYEE_A'
    assert result['normalization']['tool_message_json_payloads_decoded_locally']==1
    assert any(f['criterion']=='authorization_denied' and f['verdict']=='PASS' for f in result['findings'])
    assert store.get('evidence',e.id)==before and not store.all('run') and not store.all('usage')
