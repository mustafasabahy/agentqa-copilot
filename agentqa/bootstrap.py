from pathlib import Path
from .models import Agent, AgentProfile, Assertion, Evidence, Mode, Risk, Suite, TestCase, uid
from .adapters import DemoAdapter
from .discovery import source_profile, REFERENCE
from .generation import generate


def demo_suite(agent):
    cases=[]
    scenarios=[('WIDGET stock','How many WIDGET units are available?','WIDGET: 12 units','WIDGET',12),('GADGET zero stock','How many GADGET units are available?','GADGET: 0 units','GADGET',0),('Missing SKU','How many units are available?','Which SKU: WIDGET or GADGET?',None,None)]
    for name,question,final,sku,qty in scenarios:
        calls=[{'name':'warehouse_lookup','arguments':{'sku':sku},'result':{'quantity':qty}}] if sku else []
        assertions=[Assertion(kind='equals',path='final',expected=final)]
        if sku:
            assertions += [Assertion(kind='tool_selection',path='tool_calls.0.name',expected='warehouse_lookup'),Assertion(kind='tool_arguments',path='tool_calls.0.arguments.sku',expected=sku),Assertion(kind='calculation',path='tool_calls.0.result.quantity',expected=qty)]
        else: assertions += [Assertion(kind='count',path='tool_calls',expected=0)]
        cases.append(TestCase(name=name,objective=name,capability='stock',category='functional' if sku else 'missing_arguments',turns=[question],expected_behavior=final,assertions=assertions,evidence_requirements=['final','tool_calls'],risk=Risk.READ,modes=list(Mode),mock_evidence={'final':final,'tool_calls':calls},review_notes='Controlled demo fixture; review before live execution.'))
    cases.append(TestCase(name='Conversation memory',objective='Retain token within one session',capability='stock',category='multi_turn',turns=['Remember violet','Recall'],expected_behavior='Recall violet from same session',assertions=[Assertion(kind='equals',path='final',expected='violet'),Assertion(kind='equals',path='turns.0.final',expected='Remembered violet')],evidence_requirements=['final','turns'],risk=Risk.READ,modes=[Mode.MOCK,Mode.MULTI,Mode.STATIC,Mode.HISTORICAL,Mode.GENERATE],mock_evidence={'final':'violet','tool_calls':[],'turns':[{'final':'Remembered violet'},{'final':'violet'}]}))
    cases.append(TestCase(name='Fresh thread isolation',objective='New sessions must not inherit prior memory',capability='stock',category='thread_isolation',turns=['Recall'],expected_behavior='No memory',assertions=[Assertion(kind='thread_isolation',path='final',expected='No memory')],evidence_requirements=['final'],risk=Risk.READ,modes=list(Mode),mock_evidence={'final':'No memory','tool_calls':[]}))
    return Suite(agent_id=agent.id,name='Warehouse acceptance suite',cases=cases,generation='Hand-authored controlled fictional demo')

async def seed(store):
    if not store.all('agent'):
        agent=Agent(id='warehouse-demo',name='Warehouse · Fictional demo',adapter='demo',environment='isolated',isolation_verified=True,connection_status='controlled demo',profile=await DemoAdapter().discover())
        store.put('agent',agent); store.put('suite',demo_suite(agent))
        if REFERENCE.exists():
            hr=Agent(id='company-source',name='Company / HR · Read-only reference',adapter='source',environment='unknown',profile=source_profile(),connection_status='source only; runtime unverified')
            store.put('agent',hr); store.put('suite',generate(hr,40))
    history=Path(__file__).resolve().parent.parent/'fixtures/hr_history.json'
    if history.exists() and not store.all('evidence'):
        import json
        for record in json.loads(history.read_text()): store.put('evidence',record)
    if not store.all('golden'):
        import json
        source=Path(__file__).resolve().parent/'resources/golden.json'
        for item in json.loads(source.read_text()):store.put('golden',item)
