"""Controlled fixtures use the real LangGraph tools/persistence. No provider or target adapter."""
import hashlib
import json
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langsmith import tracing_context
from langchain_core.callbacks import BaseCallbackHandler
from .models import Agent,now,uid
from .generation import generate
from .draft_workflow import DraftCase,request_id,DraftRequest
from . import llm

class InputProbe(BaseCallbackHandler):
    def __init__(self):self.bounds=[]
    def on_chat_model_start(self,serialized,messages,**kwargs):self.bounds.append(len(str(messages).encode('utf-8'))+12000)

class FixtureModel(FakeMessagesListChatModel):
    def bind_tools(self,*args,**kwargs):return self

def fixture_cases(agent,count):
    templates=generate(agent,max(count,3)).cases
    result=[]
    for i in range(count):
        case=templates[i%len(templates)].model_copy(deep=True)
        case.name=f'Controlled fixture {i+1}: '+case.name[:140]
        case.turns=[f'Controlled scenario {hashlib.sha256(str(i).encode()).hexdigest()}: '+case.turns[0]]+case.turns[1:]
        case.reviewed=False
        result.append(DraftCase.from_case(case))
    return result

async def run(store,agent_id,actor,count=3,scenario='success'):
    DraftRequest(count=count)
    a=Agent.model_validate(store.get('agent',agent_id));invocation=uid();job=request_id(agent_id,actor,invocation)
    prompt=__import__('agentqa.provider',fromlist=['ACCEPTANCE_PROMPT']).ACCEPTANCE_PROMPT if count==40 else f'Generate {count} structured test cases for the selected agent. Do not execute tests.'
    cases=fixture_cases(a,count);prefix=[];prefix_source=None
    for recorded in store.all('copilot_evidence'):
        try:session=store.get('chat',recorded['session_id'])
        except KeyError:continue
        if session.get('mode')!='model-backed':continue
        prefix=[{'name':event['name'],'args':event['arguments'],'id':event['call_id']} for event in recorded['events'] if event['type']=='tool_call' and event['name'] in ('agent_profile','coverage_gaps')]
        if prefix:prefix_source=recorded['id'];break
    if not prefix:prefix=[{'name':'agent_profile','args':{},'id':'fixture-profile'}]
    if count!=40:prefix=prefix[:1]
    responses=[AIMessage(content='',tool_calls=[call]) for call in prefix]

    args={'count':count,'focus':'controlled offline simulation'}
    if scenario=='malformed':args['count']='Propose 1..40 cases'
    responses.append(AIMessage(content='',tool_calls=[{'name':'propose_tests','args':args,'id':'fixture-start'}]))
    for index,offset in enumerate(range(0,count,20)):
        batch=cases[offset:offset+20]
        if scenario=='partial' and index==0:batch=batch[:max(1,len(batch)-1)]
        if scenario=='duplicate' and len(batch)>1:batch[1]=batch[0]
        if scenario=='unsupported':batch[0]=batch[0].model_copy(update={'capability':'invented_operation'})
        responses.append(AIMessage(content='',tool_calls=[{'name':'save_proposed_tests','args':{'request_id':job,'batch_index':index,'cases':[c.model_dump(mode='json') for c in batch]},'id':f'fixture-batch-{index}'}]))
        if scenario!='success':break
    responses.append(AIMessage(content=f'Fixture model claims {count} cases saved. Server persistence must independently verify this.'))
    before=len([u for u in store.all('usage') if u['category']=='provider_dispatch'])
    probe=InputProbe()
    scope=['get_leave_balance'] if count==3 and any(c.name=='get_leave_balance' for c in a.profile.capabilities) else None
    with tracing_context(enabled=False):data=await llm.chat(store,agent_id,actor,invocation+'-chat',prompt,budget=1,model_override=FixtureModel(responses=responses,callbacks=[probe]),invocation=invocation,capability_scope=scope)
    after=len([u for u in store.all('usage') if u['category']=='provider_dispatch'])
    result={'id':uid(),'at':now(),'source':'CONTROLLED OFFLINE GRAPH SIMULATION — not a real model or a replay of missing historical arguments','recorded_prefix_source':prefix_source,'recorded_tool_calls_replayed':[call['name'] for call in prefix] if prefix_source else [],'historical_input':'Exact recorded 40-case user prompt reused when count=40; missing failed model output replaced with explicitly labeled fixtures','scenario':scenario,'simulated_model_steps':len(probe.bounds),'conservative_input_byte_bounds':probe.bounds,'requested_count':count,'provider_calls':after-before,'target_calls':0,'langsmith_requests':0,'result':data}
    store.put('offline_demo',result);return result
