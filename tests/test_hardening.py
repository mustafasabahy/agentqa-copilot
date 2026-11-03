import asyncio
import pytest
from agentqa.models import Mode, Evidence, Assertion, TestCase
from agentqa.policy import Policy
from agentqa.runner import Runner
from agentqa.adapters import DemoAdapter, pick
from agentqa import llm
from agentqa.security import redact


def test_usage_tokens_are_not_credentials():
    result=redact({'total_tokens':14,'input_tokens':10,'access_token':'secret','token':'secret'})
    assert result['total_tokens']==14 and result['input_tokens']==10 and result['access_token']=='[REDACTED]' and result['token']=='[REDACTED]'


def test_latest_business_call_selector_skips_metadata():
    calls=[{'name':'describe_operation','arguments':{'operation':'stock'}},{'name':'warehouse_operations','arguments':{'operation':'stock','arguments':{'sku':'WIDGET'}}},{'name':'warehouse_operations','arguments':{'operation':'stock','arguments':{'sku':'GADGET'}}}]
    assert pick({'tool_calls':calls},'tool_calls[operation=stock].arguments.arguments.sku')=='GADGET'

async def test_call_ceiling_prevents_fifth_model_request():
    callback=llm.CallCeiling(4)
    for _ in range(4):callback.on_chat_model_start()
    with pytest.raises(ValueError,match='ceiling'):callback.on_chat_model_start()

async def test_cancel_queued_run_not_left_running(setup):
    store,a,s=setup;p=Policy(store).propose(a,s,[s.cases[0].id],Mode.MOCK);runner=Runner(store)
    run=runner.start(p.id,'human');runner.cancel(run.id,'human')
    await asyncio.gather(*runner.tasks.values(),return_exceptions=True)
    assert store.get('run',run.id)['status']=='cancelled'

async def test_multi_turn_partial_evidence_persisted(setup,monkeypatch):
    store,a,s=setup;original=DemoAdapter.invoke
    async def slow(self,message,session):
        if message=='Recall':await asyncio.sleep(10)
        return await original(self,message,session)
    monkeypatch.setattr(DemoAdapter,'invoke',slow)
    p=Policy(store).propose(a,s,[s.cases[3].id],Mode.MULTI,rate_per_second=20);Policy(store).approve(p.id,'human',p.digest)
    runner=Runner(store);r=runner.start(p.id,'human');await asyncio.sleep(.15);runner.cancel(r.id,'human')
    await asyncio.gather(*runner.tasks.values(),return_exceptions=True)
    partial=store.all('partial')[0];assert partial['turns'][0]['final']=='Remembered violet'

async def test_unknown_cost_not_reported_as_measured_total(setup,monkeypatch):
    store,a,s=setup;original=DemoAdapter.invoke
    async def unknown(self,message,session):
        evidence=await original(self,message,session)
        if 'GADGET' in message:evidence.usage={}
        return evidence
    monkeypatch.setattr(DemoAdapter,'invoke',unknown)
    p=Policy(store).propose(a,s,[s.cases[0].id,s.cases[1].id],Mode.LIVE,rate_per_second=20);Policy(store).approve(p.id,'human',p.digest)
    runner=Runner(store);r=runner.start(p.id,'human');await runner.tasks[r.id]
    assert store.get('run',r.id)['measured_cost'] is None

async def test_historical_plan_binds_imported_evidence(setup):
    store,a,s=setup;e=Evidence(source='historical',final='WIDGET: 12 units',tool_calls=[]);store.put('evidence',e)
    s.cases[0].historical_evidence_id=e.id;store.put('suite',s)
    p=Policy(store).propose(a,s,[s.cases[0].id],Mode.HISTORICAL)
    assert p.historical_snapshots[e.id]['final']=='WIDGET: 12 units'
    e.final='changed';store.put('evidence',e)
    with pytest.raises(ValueError,match='evidence changed'):Policy(store).authorize(p,'human')
