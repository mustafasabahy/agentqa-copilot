"""Local parsing/grounding verification shared by model judging and offline fixtures."""
import json
from .models import JudgeOutput,Verdict
from .adapters import MISSING,pick

DIMENSIONS=['Intent understanding','Tool selection correctness','Argument extraction','Tool-result interpretation','Response grounding','Completeness','Context relevance','Multi-turn consistency','Authorization behavior','Unnecessary or redundant tool calls']

def parse_output(value,require_dimensions=False):
    if hasattr(value,'model_dump'):value=value.model_dump(mode='json')
    output=JudgeOutput.model_validate_json(value if isinstance(value,str) else json.dumps(value,allow_nan=False),strict=True)
    criteria=[f.criterion for f in output.findings]
    if any(not c.strip() for c in criteria) or len(set(criteria))!=len(criteria):raise ValueError('Judge criteria must be nonempty and unique')
    if require_dimensions and set(criteria)!=set(DIMENSIONS):raise ValueError('Judge must return exactly the ten documented dimensions')
    return output

def ground(output,evidence):
    data=evidence.model_dump(mode='json')
    tool_dimensions={DIMENSIONS[i] for i in (1,2,3,8,9)}
    for finding in output.findings:
        invalid=not finding.evidence_refs or any(pick(data,ref,MISSING) is MISSING or (pick(data,ref,MISSING) is None and not (ref.startswith('tool_calls.') and '.arguments.' in ref)) for ref in finding.evidence_refs)
        missing_tools=finding.criterion in tool_dimensions and evidence.tool_calls is None
        missing_history=finding.criterion==DIMENSIONS[7] and len(evidence.turns)<2 and len(evidence.raw.get('conversation_history',[]))<2
        if finding.verdict in (Verdict.PASS,Verdict.FAIL) and (invalid or missing_tools or missing_history or finding.uncertainty):
            finding.verdict=Verdict.INCONCLUSIVE;finding.uncertainty=True
            finding.explanation+=' [Insufficient referenced observations; server downgraded to INCONCLUSIVE.]'
    return output

def fixture(case,evidence):
    value={'findings':[{'test_case_id':case.id,'criterion':d,'score':1.0,'verdict':'PASS','evidence_refs':['final'],'explanation':'Controlled fixture assertion; not a real judge response','uncertainty':False,'improvement':'Review missing observations','model':'controlled-fixture','rubric_version':'fixture'} for d in DIMENSIONS]}
    parsed=parse_output(value,require_dimensions=True);ground(parsed,evidence)
    return {'source':'Controlled local judge-schema fixture; no model call','schema_valid':True,'dimensions':len(parsed.findings),'findings':[f.model_dump(mode='json') for f in parsed.findings],'model_calls':0}
