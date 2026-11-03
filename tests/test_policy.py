from datetime import datetime, timedelta, timezone
import pytest
from agentqa.models import Agent, Mode, Plan, Risk, Suite
from agentqa.policy import Policy, digest
from agentqa.security import PolicyError


def propose(setup,**kwargs):
    store,a,s=setup
    return Policy(store).propose(a,s,[s.cases[0].id],kwargs.pop('mode',Mode.LIVE),**kwargs)

async def test_live_requires_explicit_approval(setup):
    store,a,s=setup;p=propose(setup)
    with pytest.raises(PolicyError,match='approval'): Policy(store).authorize(p,'human')
    Policy(store).approve(p.id,'human',p.digest)
    assert Policy(store).authorize(p,'human')['digest']==p.digest
    with pytest.raises(PolicyError): Policy(store).authorize(p,'other')

@pytest.mark.parametrize('change',['case','endpoint','profile','budget','environment'])
async def test_material_changes_invalidate(setup,change):
    store,a,s=setup;p=propose(setup);policy=Policy(store);policy.approve(p.id,'human',p.digest)
    if change=='case': s.cases[0].turns=['Changed'];store.put('suite',s)
    elif change=='endpoint': a.endpoint='http://other';store.put('agent',a)
    elif change=='profile': a.profile.tools.append('new');store.put('agent',a)
    elif change=='environment': a.environment='production';store.put('agent',a)
    else: p.budget=42
    with pytest.raises(PolicyError): policy.authorize(p,'human')

async def test_digest_and_expiry(setup):
    store,a,s=setup;p=propose(setup);policy=Policy(store)
    with pytest.raises(PolicyError): policy.approve(p.id,'human','forged')
    approved=policy.approve(p.id,'human',p.digest)
    approved['expires_at']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat();store.put('approval',approved)
    with pytest.raises(PolicyError,match='expired'):policy.authorize(p,'human')

@pytest.mark.parametrize('risk',[Risk.MUTATE,Risk.PROPOSE,Risk.UNKNOWN])
@pytest.mark.parametrize('environment',['production','staging','unknown','isolated'])
async def test_unsafe_requires_server_verified_isolation(setup,risk,environment):
    store,a,s=setup;a.environment=environment;a.isolation_verified=False;s.cases[0].risk=risk
    store.put('agent',a);store.put('suite',s)
    with pytest.raises(PolicyError): propose(setup)

@pytest.mark.parametrize('risk',[Risk.MUTATE,Risk.PROPOSE,Risk.UNKNOWN])
async def test_additional_authorization(setup,risk):
    store,a,s=setup;s.cases[0].risk=risk;store.put('suite',s);p=propose(setup)
    with pytest.raises(PolicyError,match='Additional'):Policy(store).approve(p.id,'human',p.digest)
    assert Policy(store).approve(p.id,'human',p.digest,True)['additional_authorization']

async def test_unreviewed_rejected(setup):
    store,a,s=setup;s.cases[0].reviewed=False;store.put('suite',s)
    with pytest.raises(PolicyError,match='Review'):propose(setup)

async def test_duplicate_unknown_selection(setup):
    store,a,s=setup;policy=Policy(store)
    with pytest.raises(PolicyError):policy.propose(a,s,['unknown'],Mode.MOCK)
    with pytest.raises(PolicyError):policy.propose(a,s,[s.cases[0].id]*2,Mode.MOCK)

async def test_external_cost_acknowledgement(setup,monkeypatch):
    monkeypatch.delenv('AGENTQA_ALLOW_ESTIMATED_LIVE_COSTS',raising=False)
    store,a,s=setup;a.adapter='rest';store.put('agent',a)
    with pytest.raises(PolicyError,match='billing'):propose(setup,budget=1)

async def test_model_budget(setup):
    with pytest.raises(PolicyError,match='Evaluation budget'):propose(setup,judge=True,evaluation_budget=0)
