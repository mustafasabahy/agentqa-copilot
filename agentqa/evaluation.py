"""Evidence-first evaluation: missing observations never count as PASS."""
import json
import jsonschema
from .models import Evidence, Finding, TestCase, Verdict
from .adapters import MISSING, pick


def aggregate(verdicts):
    for status in (Verdict.FAIL,Verdict.ERROR,Verdict.INCONCLUSIVE):
        if status in verdicts: return status
    if verdicts and all(v==Verdict.SKIPPED for v in verdicts): return Verdict.SKIPPED
    return Verdict.PASS if verdicts else Verdict.INCONCLUSIVE


def evaluate(case: TestCase,evidence: Evidence):
    data=evidence.model_dump(mode='json'); findings=[]
    for path in case.evidence_requirements:
        value=pick(data,path,MISSING)
        if value is MISSING or value is None:
            findings.append(Finding(criterion='evidence:'+path,verdict=Verdict.INCONCLUSIVE,explanation='Required evidence is unavailable',evidence_refs=[path]))
    for a in case.assertions:
        path=a.path; value=pick(data,path,MISSING)
        if a.kind=='authorization_denied':
            # Only observed backend denial, not a refusal written by an LLM.
            value=pick(data,path or 'tool_calls',MISSING)
            if value is None or value is MISSING: ok=None
            else:
                calls=value if isinstance(value,list) else [value]
                codes=[pick(c,'result.code') for c in calls if isinstance(c,dict)]
                ok=any(c in ('forbidden','unauthorized','403','401',403,401) for c in codes) if any(c is not None for c in codes) else None
        elif value is MISSING or (value is None and not (a.kind in ('equals','tool_arguments') and a.expected is None and path.startswith('tool_calls.') and len(path.split('.'))>=4)): ok=None
        else:
            try:
                if a.kind=='tool_selection' and isinstance(value,list): ok=any(c.get('name')==a.expected for c in value)
                elif a.kind=='operation_routing' and isinstance(value,list): ok=any(c.get('arguments',{}).get('operation')==a.expected and isinstance(c.get('arguments',{}).get('arguments'),dict) for c in value)
                elif a.kind in ('equals','tool_selection','tool_arguments','operation_routing','error_code','state_invariant','calculation','thread_isolation'): ok=value==a.expected
                elif a.kind in ('contains',): ok=a.expected in value
                elif a.kind=='not_contains': ok=a.expected not in value
                elif a.kind=='exists': ok=value is not None
                elif a.kind=='schema': jsonschema.validate(value,a.expected); ok=True
                elif a.kind=='response_format':
                    parsed=json.loads(value) if isinstance(value,str) else value
                    jsonschema.validate(parsed,a.expected); ok=True
                elif a.kind in ('max','latency'): ok=value<=a.expected
                elif a.kind=='min': ok=value>=a.expected
                elif a.kind=='count': ok=len(value)==a.expected
                elif a.kind=='unique': ok=len(value)==len({json.dumps(v,sort_keys=True) for v in value})
                elif a.kind=='required_arguments': ok=all(k in value for k in a.expected)
                else: ok=None
            except (TypeError,ValueError,jsonschema.ValidationError): ok=False
        findings.append(Finding(criterion=a.description or a.kind,verdict=Verdict.INCONCLUSIVE if ok is None else Verdict.PASS if ok else Verdict.FAIL,explanation='Evidence unavailable' if ok is None else 'Assertion satisfied' if ok else f'Expected {a.kind} {a.expected!r}',evidence_refs=[path],security=a.security or a.kind=='authorization_denied'))
    if not findings: findings=[Finding(criterion='evaluation',verdict=Verdict.INCONCLUSIVE,explanation='No deterministic assertions or required evidence were specified')]
    return findings
