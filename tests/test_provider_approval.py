import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from agentqa import provider,remote_approval,llm
from agentqa.models import Agent
from agentqa.security import PolicyError,redact

@pytest.fixture
def batch(store,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','local-test-key-not-a-real-secret')
    a=Agent(id='hr-source-test',name='Reference',adapter='source');store.put('agent',a)
    for i in range(3):store.put('golden',{'id':str(i),'test_case':{'id':'case'+str(i)},'evidence':{'source':'historical'}})
    p=provider.propose(store,'human',a.id)
    return store,p

def approve(batch):
    store,p=batch;provider.approve(store,'human',p['id'],p['digest']);return store,p

def test_candidate_plan_bounded(batch):
    _,p=batch
    assert p['max_calls']==11 and p['maximum_estimated_cost_usd']<p['budget_usd']==1
    assert len(p['golden_ids'])==3 and p['generation']['count']==3

@pytest.mark.parametrize('mutate',['actor','binding','model','digest','expiry'])
def test_approval_rejects_scope_change(batch,mutate):
    store,p=approve(batch);actor='human';binding={'message':'Reply with OK.'}
    if mutate=='actor':actor='other'
    if mutate=='binding':binding['message']='Call HR'
    if mutate=='model':p['models']['judge']='unknown';store.put('provider_plan',p)
    if mutate=='digest':p['digest']='tampered';store.put('provider_plan',p)
    if mutate=='expiry':p['expires_at']='2000-01-01T00:00:00+00:00';p['digest']=provider.digest({k:v for k,v in p.items() if k!='digest'});store.put('provider_plan',p)
    with pytest.raises(PolicyError):
        with provider.scope(store,actor,p['id'],'connectivity',binding):pass

def test_key_alone_does_not_authorize_model(batch):
    with pytest.raises(ValueError,match='approval'):llm.model()

def test_missing_credentials_prevents_dispatch(batch,monkeypatch):
    store,p=approve(batch);monkeypatch.delenv('OPENAI_API_KEY')
    with pytest.raises(PolicyError,match='missing'):
        with provider.scope(store,'human',p['id'],'connectivity',{'message':'Reply with OK.'}):pass
    assert not store.all('usage')

def guard(batch):
    store,p=approve(batch)
    with provider.scope(store,'human',p['id'],'connectivity',{'message':'Reply with OK.'}):return provider.ProviderGuard(provider.CURRENT.get())

def test_call_limit_atomic_across_parallel_requests(batch):
    g=guard(batch)
    def dispatch(i):
        try:g.on_chat_model_start({},[['small prompt']],run_id=str(i));return True
        except PolicyError:return False
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(dispatch,range(8)))
    assert sum(results)==1 and len(batch[0].all('usage'))==1

def test_token_bound_denies_before_reservation(batch):
    g=guard(batch)
    with pytest.raises(PolicyError,match='token bound'):g.on_chat_model_start({},[['x'*64000]],run_id='large')
    assert not batch[0].all('usage')

@pytest.mark.parametrize('usage',[{}, {'input_tokens':513,'output_tokens':1},{'input_tokens':1,'output_tokens':65}])
def test_unexpected_usage_stops_whole_batch(batch,usage):
    g=guard(batch);g.on_chat_model_start({},[['small']],run_id='one')
    response=SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=usage))]])
    with pytest.raises(PolicyError,match='stopped'):g.on_llm_end(response,run_id='one')
    assert batch[0].get('provider_approval',batch[1]['id'])['stopped']

def test_reported_usage_cost_is_not_measured_billing(batch):
    g=guard(batch);g.on_chat_model_start({},[['small']],run_id='one')
    g.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata={'input_tokens':100,'output_tokens':10}))]]),run_id='one')
    u=batch[0].all('usage')[0];assert u['measured_cost'] is None and u['reported_token_cost_estimate']>0

def test_strings_and_nested_secrets_redacted(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','private-example')
    assert redact({'note':'private-example Bearer abcdef','nested':[{'api_key':'secret'}]})=={'note':'[REDACTED] Bearer [REDACTED]','nested':[{'api_key':'[REDACTED]'}]}

@pytest.mark.parametrize('name,expected',[('AuthenticationError','authentication'),('NotFoundError','unavailable'),('RateLimitError','quota')])
def test_provider_safe_diagnostics(name,expected):
    error=type(name,(Exception,),{})('a-secret-key')
    message=provider.safe_error(error)
    assert expected in message and 'a-secret-key' not in message

def test_remote_gate_separate_and_exact(store):
    p=remote_approval.propose(store,'human','projects',{})
    with pytest.raises(KeyError):
        with remote_approval.scope(store,'human',p['id'],'projects',{}):pass
    remote_approval.approve(store,'human',p['id'],p['digest'])
    with pytest.raises(PolicyError):
        with remote_approval.scope(store,'human',p['id'],'dataset',{}):pass
    with remote_approval.scope(store,'human',p['id'],'projects',{}):assert remote_approval.CURRENT_REMOTE.get()
    assert remote_approval.CURRENT_REMOTE.get() is None

def test_remote_request_limit_no_network(store,monkeypatch):
    p=remote_approval.propose(store,'human','projects',{});p['max_http_requests']=1;p['digest']=provider.digest({k:v for k,v in p.items() if k!='digest'});store.put('remote_plan',p)
    remote_approval.approve(store,'human',p['id'],p['digest'])
    monkeypatch.setattr('requests.Session.request',lambda *a,**k:'mock response')
    with remote_approval.scope(store,'human',p['id'],'projects',{}):
        session=remote_approval.ApprovedSession(remote_approval.CURRENT_REMOTE.get())
        assert session.request('GET',p['endpoint']+'/sessions',params=p['request']['query'])=='mock response'
        with pytest.raises(PolicyError,match='ceiling'):session.request('GET',p['endpoint']+'/sessions',params=p['request']['query'])


def test_unapproved_provider_plan_cannot_dispatch(batch):
    store,p=batch
    with pytest.raises(PolicyError,match='not been explicitly approved'):
        with provider.scope(store,'human',p['id'],'connectivity',{'message':'Reply with OK.'}):pass
    assert not store.all('usage')


def test_provider_operation_cannot_be_replayed_as_new_interaction(batch):
    store,p=approve(batch)
    with provider.scope(store,'human',p['id'],'connectivity',{'message':'Reply with OK.'}):
        g=provider.ProviderGuard(provider.CURRENT.get());g.on_chat_model_start({},[['small']],run_id='one')
    with provider.scope(store,'human',p['id'],'connectivity',{'message':'Reply with OK.'}):
        g=provider.ProviderGuard(provider.CURRENT.get())
        with pytest.raises(PolicyError,match='already started'):g.on_chat_model_start({},[['small']],run_id='two')

def test_tracing_requires_independent_remote_approval(batch):
    store,old=batch
    p=provider.propose(store,'human',old['agent_id'],trace=True);provider.approve(store,'human',p['id'],p['digest'])
    assert p['remote_trace_plan_id'] and p['trace']
    with pytest.raises(KeyError):
        with provider.scope(store,'human',p['id'],'connectivity',{'message':'Reply with OK.'}):pass
    assert provider.CURRENT.get() is None and not store.all('usage')

async def test_target_judge_toggle_cannot_authorize_paid_requests(setup):
    from agentqa.policy import Policy
    from agentqa.runner import Runner
    store,a,s=setup
    p=Policy(store).propose(a,s,[s.cases[0].id],'mock',judge=True,evaluation_budget=1)
    with pytest.raises(PolicyError,match='separately approved'):Runner(store).start(p.id,'human')
    assert not store.all('run') and not store.all('usage')


def test_recovery_requires_stopped_batch_and_preserves_total_ceiling(batch):
    store,p=approve(batch)
    with pytest.raises(PolicyError,match='stopped'):provider.propose(store,'human',p['agent_id'],retry_of=p['id'])
    with provider.scope(store,'human',p['id'],'connectivity',{'message':'Reply with OK.'}):
        g=provider.ProviderGuard(provider.CURRENT.get());g.on_chat_model_start({},[['small']],run_id='one');g.stop()
    recovery=provider.propose(store,'human',p['agent_id'],retry_of=p['id'])
    assert recovery['max_calls']==10 and not any(o['name']=='connectivity' for o in recovery['operations'])
    assert recovery['maximum_estimated_cost_usd']+recovery['prior_batch_conservative_exposure_usd']<=1
    assert not any(a['id']==recovery['id'] for a in store.all('provider_approval'))


def test_operation_failure_stops_batch_and_preserves_counters(batch):
    store,p=approve(batch)
    with pytest.raises(ValueError,match='controlled'):
        with provider.scope(store,'human',p['id'],'connectivity',{'message':'Reply with OK.'}):
            raise ValueError('controlled operation failure')
    a=store.get('provider_approval',p['id'])
    assert a['stopped'] and a['calls']=={} and a['reserved_cost']==0
    with pytest.raises(PolicyError,match='stopped'):
        with provider.scope(store,'human',p['id'],'connectivity',{'message':'Reply with OK.'}):pass

async def test_claimed_success_without_saved_drafts_stops_paid_scope(batch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    class ToolFake(FakeMessagesListChatModel):
        def bind_tools(self,*a,**kw):return self
    store,p=approve(batch)
    binding={'agent':store.get('agent',p['agent_id']),'message':p['prompt'],'session_id':None}
    with provider.scope(store,'human',p['id'],'copilot',binding):
        result=await llm.chat(store,p['agent_id'],'human',None,p['prompt'],.2,model_override=ToolFake(responses=[AIMessage(content='Successfully saved all 40 cases!')]))
    assert result['complete'] is False and 'Successfully saved' not in result['answer']
    assert store.get('provider_approval',p['id'])['stopped']
    assert not [u for u in store.all('usage') if u['category']=='provider_dispatch']


def test_review_export_preserves_business_authorization_exactly(batch):
    store,p=batch
    p['reviewed_data']={'authorization':'Self employee or authorized HR role.'}
    p['digest']=provider.digest({k:v for k,v in p.items() if k!='digest'})
    exported=provider.reviewed_plan_export(p)
    assert exported==p and exported['reviewed_data']['authorization']=='Self employee or authorized HR role.'
    assert provider.digest({k:v for k,v in exported.items() if k!='digest'})==exported['digest']


def test_review_export_rejects_credentials_instead_of_changing_digest(batch,monkeypatch):
    _,p=batch;monkeypatch.setenv('OPENAI_API_KEY','private-credential-fixture')
    p['reviewed_data']={'note':'private-credential-fixture'};p['digest']=provider.digest({k:v for k,v in p.items() if k!='digest'})
    with pytest.raises(PolicyError,match='Credential detected'):provider.reviewed_plan_export(p)
