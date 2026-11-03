"""Immutable snapshots and server-side, single-use approval enforcement."""
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta
from .models import Agent, Mode, Plan, Risk, Suite, now, uid
from .generation import RISK_ORDER, validate_case
from .config import RESERVATION
from .security import PolicyError


def digest(data):
    if hasattr(data,'model_dump'): data=data.model_dump(mode='json',by_alias=True)
    data={k:v for k,v in data.items() if k!='digest'}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Policy:
    def __init__(self,store): self.store=store
    def propose(self,agent,suite,test_ids,mode,budget=0,concurrency=1,timeout=30,judge=False,evaluation_budget=0,rate_per_second=2):
        mode=Mode(mode)
        if suite.agent_id!=agent.id: raise PolicyError('Suite belongs to another agent')
        if len(test_ids)!=len(set(test_ids)): raise PolicyError('Duplicate test selection')
        selected=[c for c in suite.cases if c.id in test_ids]
        if len(selected)!=len(test_ids) or not selected: raise PolicyError('Unknown or empty test selection')
        if not agent.profile: raise PolicyError('Discovery required')
        for c in selected:
            validate_case(c,agent.profile)
            if mode not in c.modes: raise PolicyError(f'{c.name} does not support {mode}')
        history={}
        if mode==Mode.HISTORICAL:
            for c in selected:
                if c.historical_evidence_id:
                    try:history[c.historical_evidence_id]=self.store.get('evidence',c.historical_evidence_id)
                    except KeyError:pass
        risk=max([c.risk for c in selected],key=RISK_ORDER.get)
        calls=sum(len(c.turns) for c in selected) if mode in (Mode.LIVE,Mode.MULTI) else 0
        reserve=0 if agent.adapter=='demo' else RESERVATION
        # The approved quantity is a conservative caller-side reservation, not a measured provider bill.
        estimated=calls*reserve
        plan=Plan(agent_id=agent.id,suite_id=suite.id,test_ids=test_ids,mode=mode,environment=agent.environment,risk=risk,target_fingerprint=agent.profile.fingerprint,snapshots=selected,agent_snapshot=agent.model_dump(mode='json',by_alias=True),historical_snapshots=history,budget=budget,max_calls=calls,estimated_cost=estimated,concurrency=concurrency,timeout=timeout,judge=judge,evaluation_budget=evaluation_budget,rate_per_second=rate_per_second,side_effects=['No target invocation; local generation or evaluation of stored evidence only'] if mode not in (Mode.LIVE,Mode.MULTI) else ['Requests may create proposals or change target data; cleanup is not automatic'] if risk!=Risk.READ else ['Target executes untrusted model behavior; read-only intent is not a backend write guarantee'])
        if estimated>budget: raise PolicyError('Execution reservation exceeds budget')
        if judge and evaluation_budget<len(selected)*RESERVATION: raise PolicyError('Evaluation budget cannot reserve one judge call per selected case')
        self.check_eligibility(plan,agent)
        plan.digest=digest(plan); self.store.put('plan',plan)
        return plan
    def check_eligibility(self,plan,agent):
        if plan.mode not in (Mode.LIVE,Mode.MULTI): return
        if agent.adapter in ('source','langsmith'): raise PolicyError('This adapter cannot invoke a target')
        if agent.config.get('isolated_hr_budget_file'):
            from pathlib import Path
            import json
            proof=agent.config['isolation_proof']
            if datetime.fromisoformat(agent.config['execution_expires_at'])<datetime.now(timezone.utc):raise PolicyError('Isolated HR plan expired')
            if hashlib.sha256(Path(proof['synthetic_db_path']).read_bytes()).hexdigest()!=proof['synthetic_db_sha256']:raise PolicyError('Synthetic fixture changed')
            for path,h in proof['harness_sha256'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise PolicyError('Isolated harness changed')
            for rel,h in proof['source_sha256'].items():
                if hashlib.sha256((Path(proof['source_path'])/rel).read_bytes()).hexdigest()!=h:raise PolicyError('Isolated HR source changed')

        if not all(c.reviewed for c in plan.snapshots): raise PolicyError('Review every selected case before live execution')
        if plan.mode==Mode.MULTI and not agent.profile.conversation: raise PolicyError('Target does not declare conversation support')
        if plan.mode==Mode.LIVE and any(len(c.turns)>1 for c in plan.snapshots): raise PolicyError('Multi-turn cases require multi_turn mode')
        if plan.risk in (Risk.UNKNOWN,Risk.MUTATE,Risk.PROPOSE):
            if agent.environment!='isolated' or not agent.isolation_verified: raise PolicyError('Unsafe tests require a server-verified isolated environment')
        if agent.environment in ('production','unknown') and plan.risk!=Risk.READ: raise PolicyError('Non-read-only tests forbidden against production/unknown targets')
        if agent.adapter!='demo' and not os.getenv('AGENTQA_ALLOW_ESTIMATED_LIVE_COSTS','').lower()=='true':
            raise PolicyError('External target billing is unbounded by this transport. Operator must acknowledge reservation estimates via AGENTQA_ALLOW_ESTIMATED_LIVE_COSTS=true before live eligibility')
    def approve(self,plan_id,actor,expected_digest,additional_authorization=False):
        plan=Plan.model_validate(self.store.get('plan',plan_id)); self.fresh(plan)
        agent=Agent.model_validate(self.store.get('agent',plan.agent_id)); self.check_eligibility(plan,agent)
        if expected_digest!=plan.digest: raise PolicyError('Approval digest does not match reviewed plan')
        if plan.risk!=Risk.READ and plan.mode in (Mode.LIVE,Mode.MULTI) and not additional_authorization: raise PolicyError('Additional explicit side-effect authorization required')
        approval={'id':plan.id,'actor':actor,'digest':plan.digest,'at':now(),'expires_at':(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat(),'additional_authorization':additional_authorization,'source':'authenticated UI/API human action'}
        self.store.put('approval',approval); self.store.audit(actor,'approve_plan',approval)
        return approval
    def fresh(self,plan):
        if digest(plan)!=plan.digest: raise PolicyError('Plan was altered')
        agent=Agent.model_validate(self.store.get('agent',plan.agent_id))
        if agent.model_dump(mode='json',by_alias=True)!=plan.agent_snapshot: raise PolicyError('Agent changed; create and approve a new plan')
        for id,snapshot in plan.historical_snapshots.items():
            if self.store.get('evidence',id)!=snapshot:raise PolicyError('Imported evidence changed; create a fresh plan')
        suite=Suite.model_validate(self.store.get('suite',plan.suite_id))
        current={c.id:c.model_dump(mode='json',by_alias=True) for c in suite.cases}
        if any(current.get(c.id)!=c.model_dump(mode='json',by_alias=True) for c in plan.snapshots): raise PolicyError('Selected tests changed; new approval required')
    def authorize(self,plan,actor):
        self.fresh(plan)
        agent=Agent.model_validate(self.store.get('agent',plan.agent_id)); self.check_eligibility(plan,agent)
        if plan.mode not in (Mode.LIVE,Mode.MULTI): return None
        try: approval=self.store.get('approval',plan.id)
        except KeyError: raise PolicyError('Explicit human approval is required')
        if approval['actor']!=actor or approval['digest']!=plan.digest: raise PolicyError('Approval does not authorize this actor/plan')
        if datetime.fromisoformat(approval['expires_at'])<datetime.now(timezone.utc): raise PolicyError('Approval expired')
        return approval
