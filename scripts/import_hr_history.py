#!/usr/bin/env python3
"""Copy observable historical evidence only. Never invoke or import the HR agent."""
import hashlib
import json
from pathlib import Path
from agentqa.models import Evidence, uid
from agentqa.security import redact
root=Path(__file__).resolve().parent.parent
from agentqa.discovery import REFERENCE
reference=REFERENCE
source=reference/'MEMORY_FIX_RESULTS.json'
doc=json.loads(source.read_text()); records=[]; seen_by_thread={}
for turn in doc['turns']:
    results={r['tool_call_id']:r.get('output') for r in turn.get('tool_results',[])}
    calls=[]
    known=seen_by_thread.setdefault(turn.get('thread_id','unknown'),set())
    for call in turn.get('tool_calls',[]):
        if call.get('id') in known: continue
        known.add(call.get('id'))
        calls.append({'id':call.get('id'),'name':call.get('tool'),'arguments':{'operation':call['operation'],'arguments':call.get('arguments',{})} if call.get('operation') else call.get('arguments',{}),'result':results.get(call.get('id'))})
    trace=turn.get('langsmith',{})
    e=Evidence(id='hr-history-'+turn['label'],source='historical',session_id=turn.get('thread_id'),final=turn.get('final_response'),tool_calls=calls,trace_id=trace.get('trace_id'),trace_url=trace.get('url'),raw={'provenance':'Existing local HR result file; no new run','file':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'input':turn.get('input'),'label':turn['label'],'call_scope':'New recorded tool call IDs relative to earlier turns in this file','historical_model_calls':turn.get('completed_model_calls')})
    records.append(e.model_dump(mode='json'))
(root/'fixtures/hr_history.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
print('Copied',len(records),'historical evidence records; target calls: 0')
