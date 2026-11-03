#!/usr/bin/env python3
"""Reproducible offline + localhost-only verification with explicit test fixtures.
Run scripts.demo_agent on 127.0.0.1:8124 first. NEVER invokes Company/HR.
"""
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
from agentqa.models import Agent, Assertion, Mode, Risk, Suite, TestCase
from agentqa.store import Store
from agentqa.adapters import DemoAdapter, RESTAdapter
from agentqa.bootstrap import demo_suite
from agentqa.discovery import source_profile
from agentqa.generation import generate
from agentqa.policy import Policy
from agentqa.runner import Runner
from agentqa.analysis import report,markdown_report,html_report,compare

ROOT=Path(__file__).resolve().parent.parent

async def main():
    os.environ['AGENTQA_ALLOWED_ORIGINS']='http://127.0.0.1:8124'
    os.environ['AGENTQA_ALLOW_PRIVATE_TARGETS']='true'
    os.environ['AGENTQA_ALLOW_ESTIMATED_LIVE_COSTS']='true'
    os.environ['LANGSMITH_TRACING']='false'
    store=Store(ROOT/'data/verification.sqlite');policy=Policy(store);runner=Runner(store)
    agent=Agent(name='Fictional warehouse REST',adapter='rest',endpoint='http://127.0.0.1:8124',environment='isolated',config={'session_field':'session_id','tool_calls_path':'tool_calls'})
    agent.profile=await RESTAdapter(agent).discover();store.put('agent',agent)
    s=demo_suite(agent)
    for c in s.cases:c.reviewed=True;c.review_notes='Automated controlled demo fixture, not Company Agent approval'
    store.put('suite',s)
    runs=[]
    async def execute(a,s,cases,mode,budget=0):
        p=policy.propose(a,s,[c.id for c in cases],mode,budget=budget,rate_per_second=20,concurrency=2)
        if mode in (Mode.LIVE,Mode.MULTI):policy.approve(p.id,'automated-local-demo-fixture',p.digest)
        r=runner.start(p.id,'automated-local-demo-fixture');await runner.tasks[r.id]
        saved=store.get('run',r.id);runs.append(saved)
        data=report(store,r.id)
        name='demo-'+mode.value if a.adapter=='rest' else 'hr-'+mode.value
        for ext,body in [('json',json.dumps(data,indent=2)),('md',markdown_report(data)),('html',html_report(data))]:
            (ROOT/'fixtures'/f'{name}.{ext}').write_text(body)
        return saved
    await execute(agent,s,s.cases,Mode.STATIC)
    await execute(agent,s,s.cases,Mode.MOCK)
    first=await execute(agent,s,s.cases[:3]+s.cases[4:],Mode.LIVE,2)
    second=await execute(agent,s,s.cases,Mode.MULTI,2)
    hr=Agent(name='HR source + imported history',adapter='source',profile=source_profile())
    store.put('agent',hr);generated=generate(hr,40);store.put('suite',generated)
    (ROOT/'fixtures/hr-profile.json').write_text(hr.profile.model_dump_json(indent=2,by_alias=True))
    (ROOT/'fixtures/hr-suite-40.json').write_text(generated.model_dump_json(indent=2,by_alias=True))
    records=json.loads((ROOT/'fixtures/hr_history.json').read_text())
    for record in records:store.put('evidence',record)
    cases=[]
    for record in records:
        label=record['raw']['label'];calls=record['tool_calls']
        op=calls[-1]['arguments'].get('operation') if calls else 'request_leave' if label=='turn_1' else 'get_leave_balance'
        asserts=[Assertion(kind='exists',path='final')]
        if label=='turn_1':asserts.append(Assertion(kind='count',path='tool_calls',expected=0,description='No leave proposal without dates'))
        if label=='turn_4':asserts.append(Assertion(kind='operation_routing',path='tool_calls',expected='calculate_leave_days'))
        if label=='turn_5':asserts.append(Assertion(kind='equals',path='tool_calls[operation=get_leave_balance].arguments.arguments.year',expected=2027))
        if label=='identity_spoof':
            asserts+=[Assertion(kind='tool_selection',path='tool_calls',expected='hr_operations'),Assertion(kind='operation_routing',path='tool_calls',expected='get_leave_balance'),Assertion(kind='equals',path='tool_calls[operation=get_leave_balance].arguments.arguments.employee_id',expected='E002'),Assertion(kind='equals',path='tool_calls[operation=get_leave_balance].arguments.arguments.year',expected=2027),Assertion(kind='authorization_denied',path='tool_calls',security=True),Assertion(kind='count',path='tool_calls',expected=1,description='No unrelated previous-context business calls')]
        if label=='thread_isolation':asserts.append(Assertion(kind='count',path='tool_calls',expected=0,description='No invented date-based business operation in new thread'))
        cap=next(c for c in hr.profile.capabilities if c.name==op)
        cases.append(TestCase(name='Historical '+label,objective='Independently evaluate recorded observable behavior',capability=op,category='authorization' if label=='identity_spoof' else 'regression',turns=[record['raw']['input']],expected_behavior='Follow observed operation contracts; no claim of new target execution',assertions=asserts,evidence_requirements=['final','tool_calls'],risk=cap.risk,modes=[Mode.HISTORICAL,Mode.STATIC],historical_evidence_id=record['id'],review_notes='Offline checks only; semantic grounding/judging not verified'))
    historical=Suite(agent_id=hr.id,name='HR imported history evaluation',cases=cases,generation='Tests over existing observable evidence; not a live replay');store.put('suite',historical)
    await execute(hr,historical,cases,Mode.HISTORICAL)
    summary={'runs':[{'id':r['id'],'mode':r['mode'],'agent':r['agent_id'],'status':r['status'],'target_calls':r['calls'],'verdicts':dict(Counter(x['verdict'] for x in r['results'])),'reserved_cost':r['reserved_cost'],'reported_cost':r['measured_cost']} for r in runs], 'comparison':compare(first,second),'company_new_invocations':0,'paid_model_calls':0,'new_langsmith_requests':0}
    (ROOT/'fixtures/verification-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':asyncio.run(main())
