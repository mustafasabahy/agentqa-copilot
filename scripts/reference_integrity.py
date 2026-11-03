#!/usr/bin/env python3
"""Hash protected files read-only; exclude active runtime caches and the target venv."""
import hashlib
import json
import sys
from pathlib import Path
from agentqa.discovery import REFERENCE
root=Path(__file__).resolve().parent.parent
manifest=root/'fixtures/reference_integrity.json'
ignored={'.git','.venv','.langgraph_api','__pycache__','.pytest_cache'}
current={str(p.relative_to(REFERENCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in REFERENCE.rglob('*') if p.is_file() and not set(p.relative_to(REFERENCE).parts)&ignored}
if '--record' in sys.argv:
    if manifest.exists():raise SystemExit('Baseline exists; refusing to replace')
    manifest.write_text(json.dumps(current,indent=2));print('Recorded',len(current),'protected file hashes')
else:
    baseline=json.loads(manifest.read_text()); changed=[p for p,d in baseline.items() if current.get(p)!=d]
    added=sorted(set(current)-set(baseline))
    print(json.dumps({'protected_files':len(baseline),'changed':changed,'added':added,'unchanged':not changed and not added}))
    raise SystemExit(bool(changed or added))
