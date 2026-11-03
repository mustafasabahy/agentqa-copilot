"""Typed, bounded draft workflow. No provider or target calls are made here."""
import hashlib
import json
import re
from typing import Annotated
from pydantic import Field, StrictInt, StrictStr
from .models import Contract, Assertion, Mode, Risk, Suite, TestCase, now, uid
from .generation import validate_case, deduplicate

CaseCount=Annotated[StrictInt,Field(ge=1,le=40)]
BatchIndex=Annotated[StrictInt,Field(ge=0,le=1)]
BATCH_SIZE=20

class DraftRequest(Contract):
    count:CaseCount
    focus:StrictStr=''

class DraftAssertion(Contract):
    kind:Assertion.model_fields['kind'].annotation
    path:StrictStr
    expected_json:StrictStr=Field(description='JSON-encoded expected value, e.g. 0, null, "hr_operations", or {"employee_id":"E001"}. Never a case count instruction.')
    security:bool=False
    description:str=''
    def assertion(self):
        try:value=json.loads(self.expected_json,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Non-finite JSON')))
        except (ValueError,TypeError):raise ValueError('expected_json must encode a finite JSON value') from None
        return Assertion(kind=self.kind,path=self.path,expected=value,security=self.security,description=self.description)

class DraftCase(Contract):
    name:StrictStr=Field(min_length=1,max_length=200)
    objective:StrictStr=Field(min_length=1)
    capability:StrictStr=Field(min_length=1)
    category:StrictStr=Field(min_length=1)
    priority:Annotated[StrictInt,Field(ge=1,le=5)]=3
    preconditions:list[str]=Field(default_factory=list)
    turns:list[str]=Field(min_length=1,max_length=30)
    expected_behavior:StrictStr=Field(min_length=1)
    assertions:list[DraftAssertion]=Field(default_factory=list)
    judge_rubric:list[str]=Field(default_factory=list)
    evidence_requirements:list[str]=Field(default_factory=list)
    risk:Risk
    modes:list[Mode]=Field(min_length=1)
    def to_case(self):
        return TestCase(**self.model_dump(exclude={'assertions'}),assertions=[a.assertion() for a in self.assertions],reviewed=False)
    @classmethod
    def from_case(cls,case):
        fields=case.model_dump(mode='json',exclude={'assertions'})
        return cls(**{k:v for k,v in fields.items() if k in cls.model_fields},assertions=[DraftAssertion(kind=a.kind,path=a.path,expected_json=json.dumps(a.expected,allow_nan=False),security=a.security,description=a.description) for a in case.assertions])

class SubmitBatch(Contract):
    request_id:StrictStr
    batch_index:BatchIndex
    cases:list[DraftCase]=Field(min_length=1,max_length=BATCH_SIZE)

class GeneratedDraftBatch(Contract):
    cases:list[DraftCase]=Field(min_length=1,max_length=BATCH_SIZE)

class DraftIncomplete(ValueError):
    def __init__(self,summary):
        self.summary=summary
        super().__init__(f"Draft request {summary['status']}: saved {summary['saved_count']}/{summary['requested_count']}; inspect request {summary['id']} and issues. No retry was issued.")

def request_id(agent_id,actor,invocation):
    return 'draft-'+hashlib.sha256(json.dumps([agent_id,actor,invocation]).encode()).hexdigest()[:32]

def summarize(job):
    return {k:job[k] for k in ('id','agent_id','suite_id','requested_count','saved_count','remaining_count','status','source','batch_sizes','batches','issues')}

def start(store,agent,actor,invocation,count,focus='',source='AI-authored structured drafts'):
    req=DraftRequest(count=count,focus=focus)
    if not agent.profile:raise ValueError('Discover the selected agent before drafting')
    id=request_id(agent.id,actor,invocation)
    try:
        previous=store.get('draft_request',id)
        if previous['requested_count']!=req.count or previous['focus']!=req.focus:raise ValueError('This invocation already has a different draft request; start a new request explicitly')
        return summarize(previous)
    except KeyError:pass
    suite=Suite(agent_id=agent.id,name=f'{agent.name} · {req.count} structured drafts',cases=[],generation=f'{source}; requested {req.count}; in_progress; 0 saved')
    job={'id':id,'actor':actor,'agent_id':agent.id,'suite_id':suite.id,'invocation':invocation,'focus':req.focus,'requested_count':req.count,'saved_count':0,'remaining_count':req.count,'status':'in_progress','source':source,'batch_sizes':[min(BATCH_SIZE,req.count-i) for i in range(0,req.count,BATCH_SIZE)],'batches':[],'issues':[],'created_at':now()}
    with store.lock,store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        c.execute('INSERT INTO documents VALUES(?,?,?)',('suite',suite.id,json.dumps(suite.model_dump(mode='json'))))
        c.execute('INSERT INTO documents VALUES(?,?,?)',('draft_request',id,json.dumps(job)))
    store.audit(actor,'start_draft_request',{'request_id':id,'count':count,'source':source})
    return summarize(job)

def submit(store,agent,actor,invocation,payload):
    batch=SubmitBatch.model_validate(payload)
    with store.lock,store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute("SELECT body FROM documents WHERE kind='draft_request' AND id=?",(batch.request_id,)).fetchone()
        if not row:raise ValueError('Unknown draft request_id; call propose_tests first')
        job=json.loads(row['body'])
        if (job['actor'],job['agent_id'],job['invocation'])!=(actor,agent.id,invocation):raise ValueError('Draft request belongs to another owner, agent or invocation')
        if any(b['index']==batch.batch_index for b in job['batches']):raise ValueError('Batch already submitted; no implicit retries or duplicate writes')
        if job['status']!='in_progress':raise ValueError('Draft request is closed; no implicit retry or refill')
        if batch.batch_index!=len(job['batches']) or batch.batch_index>=len(job['batch_sizes']):raise ValueError('Submit only the next planned batch index')
        expected=job['batch_sizes'][batch.batch_index]
        if len(batch.cases)>expected:raise ValueError(f'Batch accepts at most {expected} cases; received {len(batch.cases)}')
        suite=Suite.model_validate(json.loads(c.execute("SELECT body FROM documents WHERE kind='suite' AND id=?",(job['suite_id'],)).fetchone()['body']))
        accepted=[];issues=[]
        for i,wire in enumerate(batch.cases):
            try:
                case=wire.to_case();validate_case(case,agent.profile)
                for assertion in case.assertions:
                    if assertion.kind=='count' and (type(assertion.expected) is not int or assertion.expected<0):raise ValueError('count assertions need a non-negative integer expected value')
                if len(deduplicate(suite.cases+accepted+[case]))!=len(suite.cases)+len(accepted)+1:raise ValueError('Duplicate or near-duplicate case')
                accepted.append(case)
            except ValueError as exc:issues.append({'batch':batch.batch_index,'case_index':i,'error':str(exc)[:400]})
        if len(batch.cases)!=expected:issues.append({'batch':batch.batch_index,'error':f'Expected {expected} cases; received {len(batch.cases)}. No automatic refill.'})
        suite.cases.extend(accepted);job['saved_count']=len(suite.cases);job['remaining_count']=job['requested_count']-job['saved_count'];job['issues'].extend(issues)
        job['batches'].append({'index':batch.batch_index,'expected':expected,'received':len(batch.cases),'accepted':len(accepted),'rejected':len(batch.cases)-len(accepted)})
        if issues:job['status']='partial' if suite.cases else 'failed'
        elif job['remaining_count']==0:job['status']='complete'
        suite.generation=f"{job['source']}; requested {job['requested_count']}; {job['status']}; {job['saved_count']} saved"
        c.execute("UPDATE documents SET body=? WHERE kind='suite' AND id=?",(json.dumps(suite.model_dump(mode='json')),suite.id))
        c.execute("UPDATE documents SET body=? WHERE kind='draft_request' AND id=?",(json.dumps(job),job['id']))
    store.audit(actor,'submit_draft_batch',{'request_id':job['id'],'batch':batch.batch_index,'status':job['status'],'saved':job['saved_count']})
    return summarize(job)

def close_incomplete(store,id,reason):
    with store.lock:
        job=store.get('draft_request',id)
        if job['status']=='in_progress':
            job['status']='partial' if job['saved_count'] else 'failed';job['issues'].append({'error':reason});store.put('draft_request',job)
            suite=Suite.model_validate(store.get('suite',job['suite_id']));suite.generation=f"{job['source']}; requested {job['requested_count']}; {job['status']}; {job['saved_count']} saved";store.put('suite',suite)
    return summarize(job)


def explicit_requested_count(message):
    """Extract only a single explicit digit count in a generation clause, never arbitrary numbers/ranges."""
    matches=re.findall(r'\b(?:generate|propose|create)\s+([0-9]+)\s+(?:(?:meaningful|structured|distinct|test|draft)\s+)*cases?\b',message,re.I)
    values={int(n) for n in matches}
    return next(iter(values)) if len(values)==1 else None
