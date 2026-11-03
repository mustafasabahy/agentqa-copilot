#!/usr/bin/env python3
"""Generate only the new platform's local token. Never read target credentials."""
import os
import secrets
from pathlib import Path
root=Path(__file__).resolve().parent.parent
path=root/'.env'
if path.exists():
    raise SystemExit('.env already exists. Preserve it; add/rotate AGENTQA_ADMIN_TOKEN manually if needed.')
token=secrets.token_urlsafe(36)
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f: f.write('AGENTQA_ADMIN_TOKEN='+token+'\nLANGSMITH_TRACING=false\n')
print('Created private .env. Your local login token:')
print(token)
