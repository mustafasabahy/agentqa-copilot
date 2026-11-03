import asyncio
import pytest
from agentqa.models import Mode, Verdict
from agentqa.policy import Policy
from agentqa.runner import Runner
from agentqa.adapters import DemoAdapter

async def wait(runner,id):
    if id in runner.tasks: await runner.tasks[id]

@pytest.mark.parametrize('mode',[Mode.STATIC,Mode.MOCK,Mode.GENERATE,Mode.HISTORICAL])
async def test_offline_modes(setup,mode):
    store,a,s=setup;p=Policy(store).propose(a,s,[s.cases[0].id],mode)
    runner=Runner(store);r=runner.start(p.id,'human');await wait(runner,r.id)
    saved=store.get('run',r.id)
    assert saved['status']=='completed' and saved['calls']==0
    assert saved['results'][0]['verdict']==('SKIPPED' if mode==Mode.GENERATE else 'INCONCLUSIVE' if mode in (Mode.HISTORICAL,Mode.STATIC) else 'PASS')

async def test_approved_live_demo_and_single_use(setup):
    store,a,s=setup;policy=Policy(store);p=policy.propose(a,s,[s.cases[0].id,s.cases[1].id],Mode.LIVE,rate_per_second=20)
    policy.approve(p.id,'human',p.digest);runner=Runner(store);r=runner.start(p.id,'human');await wait(runner,r.id)
    saved=store.get('run',r.id)
    assert saved['status']=='completed' and saved['calls']==2 and saved['measured_cost']==0
    assert all(x['verdict']=='PASS' and x['evidence']['source']=='live' for x in saved['results'])
    with pytest.raises(ValueError,match='consumed'):runner.start(p.id,'human')

async def test_multi_turn_session_and_isolation(setup):
    store,a,s=setup;policy=Policy(store);p=policy.propose(a,s,[s.cases[3].id,s.cases[4].id],Mode.MULTI,concurrency=2,rate_per_second=20)
    policy.approve(p.id,'human',p.digest);runner=Runner(store);r=runner.start(p.id,'human');await wait(runner,r.id)
    saved=store.get('run',r.id)
    assert all(x['verdict']=='PASS' for x in saved['results'])
    sessions=[x['evidence']['session_id'] for x in saved['results']]
    assert len(set(sessions))==2

async def test_cancel_persists_partial_results(setup,monkeypatch):
    store,a,s=setup;original=DemoAdapter.invoke
    async def slow(self,message,session):
        if 'GADGET' in message: await asyncio.sleep(10)
        return await original(self,message,session)
    monkeypatch.setattr(DemoAdapter,'invoke',slow)
    p=Policy(store).propose(a,s,[s.cases[0].id,s.cases[1].id],Mode.LIVE,rate_per_second=20)
    Policy(store).approve(p.id,'human',p.digest);runner=Runner(store);r=runner.start(p.id,'human')
    await asyncio.sleep(.15);runner.cancel(r.id,'human');await wait(runner,r.id)
    saved=store.get('run',r.id)
    assert saved['status']=='cancelled' and len(saved['results'])==1

async def test_timeout_no_retry(setup,monkeypatch):
    store,a,s=setup;calls=[]
    async def slow(self,message,session):calls.append(1);await asyncio.sleep(1)
    monkeypatch.setattr(DemoAdapter,'invoke',slow)
    p=Policy(store).propose(a,s,[s.cases[0].id],Mode.LIVE,timeout=.1);Policy(store).approve(p.id,'human',p.digest)
    runner=Runner(store);r=runner.start(p.id,'human');await wait(runner,r.id)
    saved=store.get('run',r.id)
    assert len(calls)==1 and saved['results'][0]['error_category']=='timeout'
