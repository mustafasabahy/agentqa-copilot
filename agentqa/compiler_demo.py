"""Explicit controlled scenario fixtures, never model/provider results."""
from .models import Agent,uid,now
from . import llm,scenario_compiler as compiler
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langsmith import tracing_context

class FixtureModel(FakeMessagesListChatModel):
    def bind_tools(self,*args,**kwargs):return self

async def run(store,agent_id,actor,count=3,partial=False):
    agent=Agent.model_validate(store.get('agent',agent_id));caps=agent.profile.capabilities
    async def propose(a,n,focus,constraints,index):
        result=[]
        for i in range(n):
            cap=caps[(index*20+i)%len(caps)]
            args={k:__import__('agentqa.generation',fromlist=['sample']).sample(v,k) for k,v in cap.schema_.get('properties',{}).items()}
            result.append({'operation_id':cap.name,'objective':f'Controlled scenario {index*20+i+1} for {cap.name}','user_inputs':[f'Controlled scenario {index*20+i+1} operation {cap.name} arguments '+str(args)],'arguments':args,'rationale':'Controlled fixture; no backend outcome is assumed'})
        if partial:result[-1]['operation_id']='leave_balance_lookup'
        return {'proposals':result}
    fake=FixtureModel(responses=[AIMessage(content='',tool_calls=[{'name':'generate_tests','args':{'agent_id':agent_id,'count':count,'focus':'controlled compiler fixture','constraints':[]},'id':'controlled-generate'}]),AIMessage(content='Controlled fixture finished. Inspect server-verified compilation counts.')])
    before=len([u for u in store.all('usage') if u['category']=='provider_dispatch'])
    with tracing_context(enabled=False):chat=await llm.chat(store,agent_id,actor,None,f'Generate {count} test cases.',1,model_override=fake,intent_proposer=propose)
    result={'id':uid(),'at':now(),'source':'CONTROLLED COMPILER DEMO — no real AI','chat':chat,'compilation':store.all('compiled_generation')[0],'provider_calls':len([u for u in store.all('usage') if u['category']=='provider_dispatch'])-before,'target_calls':0,'langsmith_calls':0}
    store.put('compiler_demo',result);return result
