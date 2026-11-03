"""Transparent heuristic failure analysis and immutable run comparison."""
from collections import Counter, defaultdict
import html
import json


def failure_attribution(result):
    """Classify observations by their origin; an absent judge has no uncertainty."""
    failed=[f for f in result.get('findings',[]) if f['verdict']!='PASS']
    judges=result.get('judge_findings',[])
    error=result.get('error_category') or (result.get('evidence',{}).get('error') or {}).get('category')
    criteria=[f['criterion'] for f in failed]
    if error:
        key='model_execution_error' if error.startswith(('model','provider','judge')) else 'adapter_connection_error' if error in ('infrastructure','connection','adapter','timeout') else error
        origin='model' if key=='model_execution_error' else 'adapter' if key=='adapter_connection_error' else 'runner'
    elif any(f.get('security') and f['verdict']=='FAIL' for f in failed):key='authorization';origin='deterministic'
    elif any(f['verdict']=='INCONCLUSIVE' and any(word in f['criterion'].lower() for word in ('reviewer','review','clarification_annotation')) for f in failed):key='missing_reviewer_evidence';origin='reviewer'
    elif any(f['verdict']=='INCONCLUSIVE' and 'tool' in f['criterion'].lower() for f in failed):key='missing_tool_observation';origin='observation'
    elif any(f['verdict']=='INCONCLUSIVE' for f in failed):key='missing_execution_evidence';origin='observation'
    elif any(f['verdict']=='FAIL' for f in failed):key='deterministic_assertion_failure';origin='deterministic'
    elif any(j.get('uncertainty') or j.get('verdict')=='INCONCLUSIVE' for j in judges):key='judge_uncertainty';origin='judge'
    elif any(j.get('verdict') in ('FAIL','ERROR') for j in judges):key='judge_failure' if any(j.get('verdict')=='FAIL' for j in judges) else 'model_execution_error';origin='judge'
    else:key='missing_execution_evidence';origin='observation'
    return {'category':key,'origin':origin,'judge_ran':bool(judges),'criteria':criteria,'source':result.get('evidence',{}).get('source'),'evidence_id':result.get('evidence',{}).get('id'),'evidence_refs':sorted({r for f in failed for r in f.get('evidence_refs',[])})}


def clusters(run,plan=None):
    groups=defaultdict(list)
    for result in run['results']:
        if result['verdict'] not in ('FAIL','ERROR','INCONCLUSIVE'):continue
        provenance=failure_attribution(result)
        groups[provenance['category']].append((result,provenance))
    suggestions={
        'missing_execution_evidence':'Collect the missing execution observations; do not infer a pass from response text.',
        'missing_tool_observation':'Use an adapter or historical trace that actually exposes the required tool arguments/results.',
        'missing_reviewer_evidence':'Obtain the explicit independent review/rubric evidence; do not label it as an uncertain model.',
        'judge_uncertainty':'Inspect the actual judge findings and evidence references, then calibrate against human labels.',
        'deterministic_assertion_failure':'Compare expected and observed values using the immutable case snapshot; inspect routing, arguments or context only as hypotheses.',
        'authorization':'Inspect the observed backend denial/security assertion independently from tool routing.',
        'model_execution_error':'Check safe provider diagnostics, credentials, model access and structured-output support without automatic paid retries.',
        'adapter_connection_error':'Check the allowlisted endpoint, mappings and timeout; do not retry non-idempotent calls automatically.',
        'judge_failure':'Review the actual judge explanation alongside deterministic findings.',
        'budget':'Reduce scope or obtain a new explicit budget approval.'}
    return [{'category':key,'count':len(items),'test_case_ids':[r['test_case_id'] for r,_ in items],'hypothesis':suggestions.get(key,'Inspect the captured evidence before proposing a root cause.'),'confidence':'hypothesis; not proven','evidence_refs':[r['evidence']['id'] for r,_ in items],'provenance':[p for _,p in items],'reproduce':'Use the preserved plan snapshot and trusted identity; obtain fresh approval before any live reproduction.'} for key,items in groups.items()]


def compare(before,after):
    if before['agent_id']!=after['agent_id']: raise ValueError('Regression comparison requires the same agent')
    old={r['test_case_id']:r for r in before['results']}; new={r['test_case_id']:r for r in after['results']}
    rows=[]
    for id in old.keys()|new.keys():
        a=old.get(id); b=new.get(id)
        rows.append({'test_case_id':id,'before':a['verdict'] if a else None,'after':b['verdict'] if b else None,'regression':bool(a and b and a['verdict']=='PASS' and b['verdict'] in ('FAIL','ERROR')),'latency_delta_ms':(b['evidence']['latency_ms']-a['evidence']['latency_ms']) if a and b and a['evidence']['latency_ms'] is not None and b['evidence']['latency_ms'] is not None else None})
    return {'before':before['id'],'after':after['id'],'comparable_modes':before['mode']==after['mode'],'changes':rows,'regressions':sum(r['regression'] for r in rows)}


def report(store,run_id):
    run=store.get('run',run_id); plan=store.get('plan',run['plan_id'])
    from .generation import coverage
    from .models import Agent
    coverage_data=coverage(Agent.model_validate(plan['agent_snapshot']),store.all('suite'),store.all('run'))
    return {'overrides':[o for o in store.all('override') if o['run_id']==run_id],'coverage_at_export':coverage_data,'partials':[p for p in store.all('partial') if p['run_id']==run_id],'title':'AgentQA Copilot QA report','run':run,'plan':plan,'results_summary':dict(Counter(r['verdict'] for r in run['results'])),'failures':clusters(run,plan),'coverage_scope':{'selected':len(plan['snapshots']),'completed':len(run['results']),'operations':sorted({c['capability'] for c in plan['snapshots']})},'limitations':['Static success validates contracts, not target behavior.','Mock evidence is simulated. Historical evidence is not a new target execution.','Root causes are hypotheses.','Hidden reasoning and unobserved calls are never reconstructed.','Cost reservations are estimates; measured costs are only present when reported.']}


def markdown_report(data):
    run=data['run']; plan=data['plan']
    lines=['# '+data['title'],'',f"Run: {run['id']}",f"Mode: {run['mode']} · Status: {run['status']}",f"Agent: {plan['agent_snapshot']['name']} · Environment: {plan['environment']}",f"Approval: {json.dumps(run['approval'])}",f"Reserved cost: ${run['reserved_cost']:.4f} · Measured: {run['measured_cost']}",'','## Results','']
    names={c['id']:c for c in plan['snapshots']}
    for result in run['results']:
        case=names[result['test_case_id']]
        lines += [f"### {result['verdict']}: {case['name']}",case['objective'],'Reproduction turns: '+json.dumps(case['turns'],ensure_ascii=False),'Evidence:','```json',json.dumps(result,indent=2,ensure_ascii=False),'```','']
    lines+=['## Recorded manual overrides','',json.dumps(data['overrides'],indent=2),'','## Failure hypotheses','',json.dumps(data['failures'],indent=2),'','## Limitations','']+['- '+l for l in data['limitations']]
    return '\n'.join(lines)


def html_report(data):
    return '<!doctype html><html><head><meta charset="utf-8"><title>AgentQA report</title><style>body{font:15px system-ui;max-width:1100px;margin:40px auto;padding:24px;background:#f4f6fa;color:#17233a}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body><pre>'+html.escape(markdown_report(data))+'</pre></body></html>'


def calibration(store):
    golden=store.all('golden'); reviewed=[g for g in golden if g.get('reviewed_by')]
    rows=[]
    for item in reviewed:
        predicted=item.get('judge_verdict')
        if predicted: rows.append({'id':item['id'],'reference':item['reference'],'predicted':predicted,'agreement':predicted==item['reference']})
    return {'seed_examples':len(golden),'human_reviewed':len(reviewed),'compared':len(rows),'agreement':sum(r['agreement'] for r in rows)/len(rows) if rows else None,'disagreements':[r for r in rows if not r['agreement']],'note':'Seed labels are illustrative until a human reviews them.'}
