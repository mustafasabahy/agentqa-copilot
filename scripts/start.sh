#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then .venv/bin/python scripts/create_admin.py; fi
exec .venv/bin/uvicorn agentqa.app:app --host 127.0.0.1 --port 8130 --env-file .env
