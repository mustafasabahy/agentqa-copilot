import pytest
from agentqa.store import Store
from agentqa.models import Agent, Mode, Risk
from agentqa.adapters import DemoAdapter
from agentqa.bootstrap import demo_suite

@pytest.fixture
def store(tmp_path): return Store(tmp_path/'platform.sqlite')

@pytest.fixture
async def setup(store):
    a=Agent(id='demo',name='Test warehouse',adapter='demo',environment='isolated',isolation_verified=True,profile=await DemoAdapter().discover())
    s=demo_suite(a)
    for c in s.cases: c.reviewed=True
    store.put('agent',a);store.put('suite',s)
    return store,a,s

@pytest.fixture(autouse=True)
def no_paid_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING','false')
