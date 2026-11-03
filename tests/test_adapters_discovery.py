from agentqa.discovery import REFERENCE
import pytest
import httpx
from agentqa.adapters import DemoAdapter, RESTAdapter, LangGraphAdapter, TraceAdapter, SourceAdapter
from agentqa.discovery import source_profile,fingerprint
from agentqa.models import Agent, Mode
from agentqa.security import URLPolicy, PolicyError, safe_path, redact

async def test_hr_read_only_ast_discovery():
    if not REFERENCE.exists(): pytest.skip("Optional private HR reference is not configured")
    p=source_profile();assert len(p.capabilities)==60 and len(p.tools)==11
    balance=next(c for c in p.capabilities if c.name=='get_leave_balance')
    assert balance.tool=='hr_operations'
    assert balance.schema_['properties']['year']['minimum']==1900
    assert balance.provenance=='verified' and 'runtime unverified' in balance.evidence[0]
    assert p.fingerprint==fingerprint(p)
    with pytest.raises(ValueError):source_profile('/tmp')

async def test_rest_mappings_and_sessions():
    seen=[]
    profile=await DemoAdapter().discover()
    async def handler(request):
        seen.append(request)
        if request.method=='GET':return httpx.Response(200,json=profile.model_dump(mode='json',by_alias=True))
        return httpx.Response(200,json={'answer':'12 units','calls':[{'name':'stock','arguments':{'sku':'WIDGET'}}]})
    a=Agent(name='REST',adapter='rest',endpoint='http://127.0.0.1:8124',config={'response_path':'answer','tool_calls_path':'calls','session_field':'session'})
    adapter=RESTAdapter(a,URLPolicy(['http://127.0.0.1:8124'],True),httpx.MockTransport(handler))
    p=await adapter.discover();assert p.capabilities and Mode.MULTI in p.supported_modes
    session=await adapter.create_session();e=await adapter.invoke('WIDGET',session)
    assert e.final=='12 units' and e.tool_calls[0]['name']=='stock' and e.source=='live'
    assert session.encode() in seen[-1].content

async def test_langgraph_thread_runs_observability():
    async def handler(request):
        path=request.url.path
        if path=='/assistants/search':return httpx.Response(200,json=[{'assistant_id':'agent','graph_id':'main'}])
        if path=='/assistants/agent/schemas':return httpx.Response(200,json={'input_schema':{'type':'object'}})
        if path=='/threads':return httpx.Response(200,json={'thread_id':'thread'})
        return httpx.Response(200,json={'messages':[{'type':'ai','tool_calls':[{'id':'call','name':'lookup','args':{'sku':'WIDGET'}}]},{'type':'tool','tool_call_id':'call','content':{'quantity':12}},{'type':'ai','content':'12 units'}]})
    a=Agent(name='Graph',adapter='langgraph',endpoint='http://127.0.0.1:8124',config={'assistant_id':'agent'})
    adapter=LangGraphAdapter(a,URLPolicy(['http://127.0.0.1:8124'],True),httpx.MockTransport(handler))
    p=await adapter.discover();assert p.framework=='LangGraph Agent Server' and not p.capabilities
    session=await adapter.create_session();e=await adapter.invoke('stock',session)
    assert e.final=='12 units' and e.tool_calls[0]['result']=={'quantity':12}

async def test_no_fabricated_tools():
    async def handler(request):return httpx.Response(200,json={'response':'Answer'})
    a=Agent(name='REST',adapter='rest',endpoint='http://127.0.0.1:8124')
    e=await RESTAdapter(a,URLPolicy(['http://127.0.0.1:8124'],True),httpx.MockTransport(handler)).invoke('Hi','s')
    assert e.tool_calls is None

async def test_no_redirect_following():
    async def handler(request):return httpx.Response(302,headers={'location':'http://169.254.169.254/'})
    a=Agent(name='REST',adapter='rest',endpoint='http://127.0.0.1:8124')
    with pytest.raises(ValueError,match='redirect'):await RESTAdapter(a,URLPolicy(['http://127.0.0.1:8124'],True),httpx.MockTransport(handler)).invoke('Hi','s')

@pytest.mark.parametrize('url',['file:///etc/passwd','http://127.0.0.1:9999','http://u:p@127.0.0.1:8124','http://127.0.0.1:8124/#x','https://evil.example'])
def test_ssrf_denial(url):
    with pytest.raises(PolicyError):URLPolicy(['http://127.0.0.1:8124'],False).validate(url)

def test_private_and_metadata_denial():
    with pytest.raises(PolicyError):URLPolicy(['http://127.0.0.1:8124'],False).validate('http://127.0.0.1:8124')
    with pytest.raises(PolicyError):URLPolicy(['http://169.254.169.254'],True).validate('http://169.254.169.254')

@pytest.mark.parametrize('path',['//evil/path','/../x','/x#frag','/x?url=evil','https://evil','/\\evil'])
def test_paths_cannot_change_origin(path):
    with pytest.raises(PolicyError):safe_path('http://127.0.0.1',path)

def test_redaction():assert redact({'api_key':'secret','nested':{'Authorization':'token'}})=={'api_key':'[REDACTED]','nested':{'Authorization':'[REDACTED]'}}

async def test_offline_adapters_never_invoke():
    for adapter in (SourceAdapter(),TraceAdapter()):
        with pytest.raises(ValueError):await adapter.invoke('Hello','thread')

async def test_rest_metadata_missing_risk_stays_unknown_and_hr_schema_key_supported():
    async def handler(request):return httpx.Response(200,json={'operations':{'unknown_action':{'description':'Unknown effect'},'get_employee':{'module':'employees','write':False,'arguments_schema':{'type':'object','properties':{'name':{'type':'string'}},'required':['name']}}}})
    a=Agent(name='REST',adapter='rest',endpoint='http://127.0.0.1:8124',config={'operation_tools':{'get_employee':'get_employee'}})
    profile=await RESTAdapter(a,URLPolicy(['http://127.0.0.1:8124'],True),httpx.MockTransport(handler)).discover()
    assert profile.capabilities[0].risk=='unknown'
    assert profile.capabilities[1].tool=='get_employee' and profile.capabilities[1].schema_['required']==['name']


def test_tool_result_json_is_parsed_without_inventing_data():
    from agentqa.adapters import normalize_tool_result
    assert normalize_tool_result('{"code":"forbidden"}')=={'code':'forbidden'}
    assert normalize_tool_result({'output':'{"ok":true}'})=={'ok':True}
    assert normalize_tool_result('unstructured')=='unstructured'
