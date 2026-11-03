"""Explicit, narrow external checks. Never invoke the HR target."""
import os
import pytest
from agentqa.models import Evidence
from agentqa import llm

@pytest.mark.skipif(os.getenv('AGENTQA_RUN_LLM_TESTS')!='1',reason='Paid provider check is explicitly opt-in')
async def test_configured_provider_connectivity():
    from agentqa.config import DB
    from agentqa.store import Store
    from agentqa.provider import scope
    id=os.getenv('AGENTQA_LIVE_PROVIDER_PLAN')
    if not id:pytest.skip('Requires an exact already human-approved provider plan ID')
    store=Store(DB)
    with scope(store,'local-admin',id,'connectivity',{'message':'Reply with OK.'}):
        response=await llm.model(max_tokens=64).ainvoke('Reply with OK.')
        assert response.usage_metadata and response.content

@pytest.mark.skipif(os.getenv('AGENTQA_RUN_LANGSMITH_TESTS')!='1',reason='Live LangSmith access is explicitly opt-in')
def test_configured_langsmith_read_only():
    if not os.getenv('LANGSMITH_API_KEY'):pytest.skip('Platform LangSmith credentials unavailable')
    from agentqa.langsmith_integration import projects
    from agentqa.remote_approval import scope
    from agentqa.store import Store
    from agentqa.config import DB
    id=os.getenv('AGENTQA_LIVE_REMOTE_PLAN')
    if not id:pytest.skip('Requires a separately human-approved project-list plan')
    with scope(Store(DB),'local-admin',id,'projects',{}):assert isinstance(projects(),list)
