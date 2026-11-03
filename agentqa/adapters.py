"""Agent-agnostic adapters. Evidence is only what the transport exposes."""
import asyncio
import json
import os
import time
from typing import Protocol
import httpx
from .models import Agent, AgentProfile, Capability, Evidence, Mode, Risk, uid
from .discovery import fingerprint, source_profile
from .security import URLPolicy, safe_path, redact

MISSING=object()
def pick(data, path, default=None):
    import re
    current=data
    for part in path.split('.') if path else []:
        try:
            match=re.fullmatch(r'([^\[]+)\[(name|operation)=([^\]]+)\]',part)
            if match:
                key,field,wanted=match.groups()
                items=current[key]
                matches=[item for item in items if (item.get('name') if field=='name' else item.get('arguments',{}).get('operation') if 'arguments' in item.get('arguments',{}) else None)==wanted]
                current=matches[-1]
            else: current=current[int(part)] if isinstance(current,list) else current[part]
        except (KeyError,IndexError,TypeError,ValueError,StopIteration): return default
    return current

def normalize_tool_result(value):
    if isinstance(value,dict) and set(value)=={'output'}:value=value['output']
    if isinstance(value,dict) and value.get('type')=='tool' and isinstance(value.get('content'),str):value=value['content']
    if isinstance(value,str):
        try:return json.loads(value)
        except ValueError:return value
    return value

class Adapter(Protocol):
    async def discover(self) -> AgentProfile: ...
    async def create_session(self) -> str: ...
    async def invoke(self, message: str, session: str) -> Evidence: ...
    async def cleanup(self, session: str) -> None: ...

class RESTAdapter:
    def __init__(self, agent: Agent, policy=None, transport=None):
        self.agent=agent; self.policy=policy or URLPolicy(); self.transport=transport
    def headers(self):
        headers={}
        if self.agent.auth_env:
            secret=os.getenv(self.agent.auth_env)
            if not secret: raise ValueError('Configured target secret environment variable is missing')
            header=self.agent.config.get('auth_header','Authorization')
            if header not in ('Authorization','X-Api-Key'):raise ValueError('Unsupported target authentication header')
            headers[header]=secret if self.agent.config.get('auth_scheme')=='raw' else 'Bearer '+secret
        # Only explicitly declared non-secret identity values. No caller input per invocation.
        if self.agent.config.get('demo_identity'): headers['X-Demo-User']=self.agent.config['demo_identity']
        return headers
    async def request(self, method, path, **kwargs):
        url=self.policy.validate(safe_path(self.agent.endpoint,path))
        async with httpx.AsyncClient(timeout=30,follow_redirects=False,trust_env=False,transport=self.transport) as client:
            response=await client.request(method,url,headers=self.headers(),**kwargs)
        if 300<=response.status_code<400: raise ValueError('Target redirects are forbidden')
        if len(response.content)>2_000_000: raise ValueError('Target response exceeds evidence limit')
        return response
    async def discover(self):
        # Discovery GET is restricted to a configured metadata endpoint.
        path=self.agent.config.get('discovery_path','/capabilities')
        r=await self.request('GET',path); r.raise_for_status(); data=r.json()
        if 'operations' in data:
            caps=[]
            for name,m in data['operations'].items():
                tool=self.agent.config.get('operation_tools',{}).get(name,m.get('tool'))
                risk=Risk.PROPOSE if m.get('write') is True else Risk.READ if m.get('write') is False or m.get('read_only') is True else Risk(m.get('risk','unknown'))
                caps.append(Capability(name=name,tool=tool or m.get('module','unknown')+'_operations',description=m.get('description') or '',schema=m.get('arguments_schema',m.get('input_schema',m.get('schema',{'type':'object'}))),risk=risk,provenance='verified' if m.get('tool') else 'user_declared' if tool else 'inferred',authorization=m.get('authorization','unknown'),evidence=[path]))
            p=AgentProfile(framework='REST',capabilities=caps,tools=sorted({c.tool for c in caps}),unknowns=['Tool calls unobservable unless response mapping supplies them'],evidence=[path])
        else: p=AgentProfile.model_validate(data)
        p.supported_modes=list(Mode) if self.agent.config.get('session_field') else [m for m in Mode if m!=Mode.MULTI]
        p.fingerprint=fingerprint(p); return p
    async def create_session(self): return uid()
    async def invoke(self,message,session):
        config=self.agent.config
        body={config.get('message_field','message'):message}
        if config.get('session_field'): body[config['session_field']]=session
        t=time.monotonic()
        r=await self.request('POST',config.get('invoke_path','/chat'),json=body)
        try: data=r.json()
        except ValueError: data={'response':r.text}
        final=pick(data,config.get('response_path','response'))
        calls=pick(data,config['tool_calls_path']) if config.get('tool_calls_path') else None
        if calls is not None and not isinstance(calls,list): calls=None
        return Evidence(source='live',session_id=session,final=str(final) if final is not None else None,tool_calls=redact(calls),latency_ms=(time.monotonic()-t)*1000,raw=redact(data),usage=pick(data,config.get('usage_path','usage'),{}),error={'code':str(r.status_code),'message':'Target HTTP error'} if r.is_error else None)
    async def cleanup(self,session):
        # Remote cleanup may mutate data and is not silently performed.
        return None

class LangGraphAdapter(RESTAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.seen={}
    async def discover(self):
        r=await self.request('POST','/assistants/search',json={'limit':100}); r.raise_for_status()
        assistants=r.json(); target=self.agent.config.get('assistant_id')
        assistant=next((a for a in assistants if a.get('assistant_id')==target or a.get('graph_id')==target),None)
        if not assistant: raise ValueError('Configure an assistant_id returned by /assistants/search')
        sid=assistant['assistant_id']
        schema=await self.request('GET',f'/assistants/{sid}/schemas'); schema.raise_for_status()
        # Server input schema does not reveal tool contracts. Only explicit metadata is used.
        declared=self.agent.config.get('declared_capabilities',[])
        declared=[{**c,'provenance':'user_declared'} for c in declared]
        p=AgentProfile(framework='LangGraph Agent Server',capabilities=[Capability.model_validate(c) for c in declared],conversation=True,observable_events=['messages','tool_calls','tool messages'],supported_modes=list(Mode),unknowns=['Tools not discoverable from input schema; enrich explicitly'],evidence=[json.dumps(redact(schema.json()))[:8000]])
        p.tools=sorted({c.tool for c in p.capabilities}); p.fingerprint=fingerprint(p); return p
    async def create_session(self):
        r=await self.request('POST','/threads',json={}); r.raise_for_status(); return r.json()['thread_id']
    async def invoke(self,message,session):
        t=time.monotonic()
        r=await self.request('POST',f'/threads/{session}/runs/wait',json={'assistant_id':self.agent.config['assistant_id'],'input':{'messages':[{'role':'human','content':message}]}})
        r.raise_for_status(); data=r.json(); all_messages=data.get('messages',[])
        known=self.seen.get(session,set())
        messages=[m for m in all_messages if not m.get('id') or m['id'] not in known]
        self.seen[session]=known|{m['id'] for m in all_messages if m.get('id')}
        calls=[]; results={m.get('tool_call_id'):normalize_tool_result(m.get('content')) for m in messages if m.get('type')=='tool'}
        for m in messages:
            for call in m.get('tool_calls',[]): calls.append({'name':call.get('name'),'arguments':call.get('args',{}),'result':results.get(call.get('id')),'id':call.get('id')})
        ai=[m for m in messages if m.get('type') in ('ai','assistant') and m.get('content')]
        usage={}
        if self.agent.config.get('isolated_hr_budget_file'):
            from .store import Store
            from .config import DB
            from pathlib import Path
            active=json.loads(Path(self.agent.config['isolated_hr_budget_file']).read_text())
            records=[u for u in Store(DB).all('isolated_hr_usage') if u['plan_id']==active['plan_id']]
            turn=max((u['target_turn'] for u in records),default=0)
            current=[u for u in records if u['target_turn']==turn and u['status']=='completed']
            usage={k:sum(u['usage'][k] for u in current) for k in ('input_tokens','output_tokens','total_tokens')}
            usage['estimated_cost']=sum(u['estimated_cost'] for u in current)
            usage['model_calls']=len(current)
        return Evidence(source='live',session_id=session,final=str(ai[-1]['content']) if ai else None,tool_calls=redact(calls),events=redact(messages),latency_ms=(time.monotonic()-t)*1000,raw=redact(data),usage=usage)

class DemoAdapter:
    """Controlled fictional warehouse agent; no model or business system calls."""
    def __init__(self,agent=None): self.sessions={}
    async def discover(self):
        caps=[Capability(name='stock',tool='warehouse_lookup',description='Read fictional stock by SKU. Supported WIDGET and GADGET.',schema={'type':'object','properties':{'sku':{'type':'string','enum':['WIDGET','GADGET']}},'required':['sku'],'additionalProperties':False},risk=Risk.READ,provenance='verified',evidence=['built-in controlled demo implementation'])]
        p=AgentProfile(framework='Fictional warehouse REST',capabilities=caps,tools=['warehouse_lookup'],conversation=True,observable_events=['final','tool_calls'],supported_modes=list(Mode))
        p.fingerprint=fingerprint(p); return p
    async def create_session(self):
        s=uid(); self.sessions[s]={}; return s
    async def invoke(self,message,session):
        if session not in self.sessions: raise ValueError('Unknown demo session')
        text=message.upper(); sku=next((s for s in ['WIDGET','GADGET'] if s in text),None)
        if 'REMEMBER' in text:
            self.sessions[session]['last']=message.split()[-1]
            return Evidence(source='live',session_id=session,final='Remembered '+self.sessions[session]['last'],tool_calls=[],latency_ms=0,usage={'cost':0})
        if 'RECALL' in text: return Evidence(source='live',session_id=session,final=self.sessions[session].get('last','No memory'),tool_calls=[],latency_ms=0,usage={'cost':0})
        if not sku: return Evidence(source='live',session_id=session,final='Which SKU: WIDGET or GADGET?',tool_calls=[],latency_ms=0,usage={'cost':0})
        qty={'WIDGET':12,'GADGET':0}[sku]
        return Evidence(source='live',session_id=session,final=f'{sku}: {qty} units',tool_calls=[{'name':'warehouse_lookup','arguments':{'sku':sku},'result':{'quantity':qty}}],latency_ms=0,usage={'cost':0})
    async def cleanup(self,session): self.sessions.pop(session,None)

class SourceAdapter:
    async def discover(self): return source_profile()
    async def create_session(self): raise ValueError('Source-only target cannot execute live')
    async def invoke(self,message,session): raise ValueError('Source-only target cannot execute live')
    async def cleanup(self,session): pass

class TraceAdapter:
    async def discover(self): return AgentProfile(framework='Historical LangSmith traces',supported_modes=[Mode.HISTORICAL],unknowns=['No target invocation; profile must be enriched from observed traces'])
    async def create_session(self): raise ValueError('Historical adapter cannot invoke targets')
    async def invoke(self,message,session): raise ValueError('Historical adapter cannot invoke targets')
    async def cleanup(self,session): pass


def adapter_for(agent):
    return {'demo':lambda:DemoAdapter(agent),'source':SourceAdapter,'rest':lambda:RESTAdapter(agent),'langgraph':lambda:LangGraphAdapter(agent),'langsmith':TraceAdapter}[agent.adapter]()
