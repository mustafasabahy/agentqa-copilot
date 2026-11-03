from __future__ import annotations
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator


def uid(): return str(uuid4())
def now(): return datetime.now(timezone.utc).isoformat()

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Risk(StrEnum):
    READ = 'read_only'
    PROPOSE = 'propose_only'
    MUTATE = 'mutating'
    UNKNOWN = 'unknown'

class Verdict(StrEnum):
    PASS = 'PASS'
    FAIL = 'FAIL'
    INCONCLUSIVE = 'INCONCLUSIVE'
    ERROR = 'ERROR'
    SKIPPED = 'SKIPPED'

class Mode(StrEnum):
    GENERATE = 'generate_only'
    STATIC = 'static'
    HISTORICAL = 'historical'
    MOCK = 'mock'
    LIVE = 'live'
    MULTI = 'multi_turn'

class Capability(Contract):
    name: str
    tool: str
    description: str = ''
    schema_: dict = Field(default_factory=lambda: {'type':'object'}, alias='schema')
    risk: Risk = Risk.UNKNOWN
    provenance: Literal['verified','inferred','user_declared','unknown'] = 'unknown'
    evidence: list[str] = Field(default_factory=list)
    authorization: str = 'unknown'

class AgentProfile(Contract):
    framework: str = 'unknown'
    capabilities: list[Capability] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    model_metadata: dict = Field(default_factory=dict)
    conversation: bool = False
    observable_events: list[str] = Field(default_factory=list)
    supported_modes: list[Mode] = Field(default_factory=lambda:[Mode.GENERATE,Mode.STATIC,Mode.MOCK])
    unknowns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    fingerprint: str = ''

class Agent(Contract):
    id: str = Field(default_factory=uid)
    name: str = Field(min_length=1,max_length=160)
    adapter: Literal['rest','langgraph','langsmith','demo','source']
    endpoint: str = ''
    environment: Literal['production','staging','isolated','unknown'] = 'unknown'
    config: dict = Field(default_factory=dict)
    auth_env: str | None = None
    # Only server configuration/verification endpoint may set these fields.
    isolation_verified: bool = False
    connection_status: str = 'unverified'
    last_discovery: str | None = None
    profile: AgentProfile | None = None

class Assertion(Contract):
    kind: Literal['equals','contains','not_contains','exists','schema','max','min','count','unique','authorization_denied','tool_selection','tool_arguments','required_arguments','operation_routing','error_code','state_invariant','response_format','latency','calculation','thread_isolation']
    path: str = ''
    expected: Any = None
    security: bool = False
    description: str = ''

class TestCase(Contract):
    __test__ = False
    id: str = Field(default_factory=uid)
    name: str = Field(min_length=1,max_length=200)
    description: str = ''
    objective: str = Field(min_length=1)
    capability: str
    category: str
    priority: int = Field(default=3,ge=1,le=5)
    preconditions: list[str] = Field(default_factory=list)
    turns: list[str] = Field(min_length=1,max_length=30)
    expected_behavior: str
    assertions: list[Assertion] = Field(default_factory=list)
    judge_rubric: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    risk: Risk = Risk.UNKNOWN
    modes: list[Mode] = Field(default_factory=lambda:[Mode.STATIC,Mode.MOCK])
    estimated_cost: float = Field(default=0,ge=0,allow_inf_nan=False)
    cleanup: list[str] = Field(default_factory=list)
    reviewed: bool = False
    review_notes: str = ''
    # Explicit controlled fixture, never presented as real target output.
    mock_evidence: dict | None = None
    historical_evidence_id: str | None = None

class Suite(Contract):
    id: str = Field(default_factory=uid)
    agent_id: str
    name: str
    cases: list[TestCase]
    created_at: str = Field(default_factory=now)
    generation: str = 'deterministic'

class Plan(Contract):
    id: str = Field(default_factory=uid)
    agent_id: str
    suite_id: str
    test_ids: list[str] = Field(min_length=1)
    mode: Mode
    environment: str
    risk: Risk
    target_fingerprint: str
    snapshots: list[TestCase]
    agent_snapshot: dict
    historical_snapshots: dict[str,dict] = Field(default_factory=dict)
    budget: float = Field(ge=0,le=10000,allow_inf_nan=False)
    max_calls: int = Field(ge=0,le=10000)
    estimated_cost: float = Field(ge=0,allow_inf_nan=False)
    timeout: float = Field(default=30,ge=0.1,le=300)
    concurrency: int = Field(default=1,ge=1,le=8)
    rate_per_second: float = Field(default=2,gt=0,le=20)
    judge: bool = False
    evaluation_budget: float = Field(default=0,ge=0,le=1000,allow_inf_nan=False)
    side_effects: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now)
    digest: str = ''

class Evidence(Contract):
    id: str = Field(default_factory=uid)
    source: Literal['live','mock','historical','static']
    session_id: str | None = None
    final: str | None = None
    tool_calls: list[dict] | None = None
    events: list[dict] = Field(default_factory=list)
    turns: list[dict] = Field(default_factory=list)
    latency_ms: float | None = None
    usage: dict = Field(default_factory=dict)
    state_before: Any = None
    state_after: Any = None
    error: dict | None = None
    trace_id: str | None = None
    trace_url: str | None = None
    raw: dict = Field(default_factory=dict)

class Finding(Contract):
    criterion: str
    verdict: Verdict
    explanation: str
    evidence_refs: list[str] = Field(default_factory=list)
    security: bool = False

class JudgeFinding(Contract):
    test_case_id: str
    criterion: str
    score: float = Field(ge=0,le=1)
    verdict: Verdict
    evidence_refs: list[str]
    explanation: str
    uncertainty: bool
    improvement: str
    model: str
    rubric_version: str

class JudgeOutput(Contract):
    findings: list[JudgeFinding] = Field(min_length=1)

class Result(Contract):
    id: str = Field(default_factory=uid)
    test_case_id: str
    verdict: Verdict
    findings: list[Finding] = Field(default_factory=list)
    judge_findings: list[JudgeFinding] = Field(default_factory=list)
    evidence: Evidence
    error_category: str | None = None

class Run(Contract):
    id: str = Field(default_factory=uid)
    plan_id: str
    agent_id: str
    suite_id: str
    mode: Mode
    status: str = 'queued'
    results: list[Result] = Field(default_factory=list)
    started_at: str = Field(default_factory=now)
    ended_at: str | None = None
    reserved_cost: float = 0
    measured_cost: float | None = None
    calls: int = 0
    cancellation_requested: bool = False
    approval: dict | None = None
    reproducibility: dict = Field(default_factory=dict)
