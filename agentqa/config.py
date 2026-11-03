import os
from pathlib import Path
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parent.parent
load_dotenv(ROOT/".env")
DB=os.getenv('AGENTQA_DB',str(ROOT/'data/platform.sqlite'))
MODEL=os.getenv('AGENTQA_MODEL','openai:gpt-5.4-mini-2026-03-17')
ROLE_MODELS={role:os.getenv('AGENTQA_'+role.upper()+'_MODEL',MODEL) for role in ('copilot','generation','judge')}
RESERVATION=float(os.getenv('AGENTQA_CALL_RESERVATION','0.25'))
MAX_TOKENS=int(os.getenv('AGENTQA_LLM_MAX_TOKENS','2048'))

import math
if not math.isfinite(RESERVATION) or RESERVATION<=0:raise ValueError('AGENTQA_CALL_RESERVATION must be finite and positive')
if not 1<=MAX_TOKENS<=65536:raise ValueError('AGENTQA_LLM_MAX_TOKENS must be 1..65536')

GENERATION_MAX_TOKENS=int(os.getenv('AGENTQA_GENERATION_MAX_TOKENS','12000'))
if not 1<=GENERATION_MAX_TOKENS<=65536:raise ValueError('AGENTQA_GENERATION_MAX_TOKENS must be 1..65536')
