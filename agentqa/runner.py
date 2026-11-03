import asyncio
import platform
import time
import jsonschema
from .models import Agent, Evidence, Finding, Mode, Plan, Result, Run, Verdict, now
from .adapters import adapter_for
from .evaluation import evaluate, aggregate
from .policy import Policy
from .config import RESERVATION

class Runner:
    def __init__(self,store):
        self.store=store; self.policy=Policy(store); self.tasks={}; self.locks={}
    def start(self,plan_id,actor):
        plan=Plan.model_validate(self.store.get('plan',plan_id))
        if plan.judge:
            from .security import PolicyError
            raise PolicyError('Run deterministic evaluation first. Judge recorded examples through a separately approved provider scope; target approval cannot authorize paid judging.')
        approval=self.policy.authorize(plan,actor)
        run=Run(plan_id=plan.id,agent_id=plan.agent_id,suite_id=plan.suite_id,mode=plan.mode,approval=approval,reproducibility={'python':platform.python_version(),'plan_digest':plan.digest,'agent_fingerprint':plan.target_fingerprint,'retries':0,'cleanup':'Adapter-owned session cleanup only; business rollback unsupported'})
        self.store.claim(plan.id,run.id); self.store.put('run',run)
        self.tasks[run.id]=asyncio.create_task(self.execute(run,plan,actor))
        return run
    def cancel(self,id,actor):
        run=Run.model_validate(self.store.get('run',id)); run.cancellation_requested=True; run.status='cancelled'; run.ended_at=now(); self.store.put('run',run)
        self.store.audit(actor,'cancel_run',{'run_id':id})
        if id in self.tasks: self.tasks[id].cancel()
        return run
    async def execute(self,run,plan,actor):
        agent=Agent.model_validate(plan.agent_snapshot); adapter=adapter_for(agent)
        semaphore=asyncio.Semaphore(plan.concurrency); rate_lock=asyncio.Lock(); budget_lock=asyncio.Lock()
        next_request=0.; eval_spent=0.; cost_unknown=False
        run.status='running'; self.store.put('run',run)
        async def invoke(message,session):
            nonlocal next_request
            async with rate_lock:
                delay=max(0,next_request-time.monotonic())
                if delay: await asyncio.sleep(delay)
                next_request=time.monotonic()+1/plan.rate_per_second
            async with budget_lock:
                reserve=0 if agent.adapter=='demo' else RESERVATION
                if run.calls+1>plan.max_calls or run.reserved_cost+reserve>plan.budget+1e-9: raise ValueError('Execution budget/call ceiling reached')
                self.policy.authorize(plan,actor)  # Recheck integrity immediately before each target turn.
                run.calls+=1; run.reserved_cost+=reserve; self.store.put('run',run)
            return await asyncio.wait_for(adapter.invoke(message,session),plan.timeout)
        async def one(case):
            nonlocal eval_spent, cost_unknown
            async with semaphore:
                evidence=Evidence(source='static'); session=None; error_category=None
                try:
                    if plan.mode==Mode.GENERATE:
                        result=Result(test_case_id=case.id,verdict=Verdict.SKIPPED,evidence=evidence,findings=[Finding(criterion='generate_only',verdict=Verdict.SKIPPED,explanation='Case generated; no execution or behavior evaluation')])
                    elif plan.mode==Mode.STATIC:
                        cap=next(c for c in agent.profile.capabilities if c.name==case.capability)
                        jsonschema.Draft202012Validator.check_schema(cap.schema_)
                        result=Result(test_case_id=case.id,verdict=Verdict.INCONCLUSIVE,evidence=evidence,findings=[Finding(criterion='static_contract',verdict=Verdict.PASS,explanation='Capability and JSON schema structure valid.'),Finding(criterion='behavioral_evidence',verdict=Verdict.INCONCLUSIVE,explanation='Static validation does not execute the target or supply the behavioral evidence required by this case.')])
                    else:
                        if plan.mode==Mode.MOCK:
                            evidence=Evidence.model_validate({'source':'mock',**(case.mock_evidence or {})})
                            evidence.source='mock'
                        elif plan.mode==Mode.HISTORICAL:
                            if not case.historical_evidence_id: raise LookupError('Select an imported historical evidence record for this case')
                            evidence=Evidence.model_validate(plan.historical_snapshots[case.historical_evidence_id]); evidence.source='historical'
                        else:
                            session=await asyncio.wait_for(adapter.create_session(),plan.timeout)
                            observed=[]; started=time.monotonic()
                            for message in case.turns:
                                turn=await invoke(message,session); observed.append(turn.model_dump(mode='json'))
                                self.store.put('partial',{'id':run.id+'-'+case.id,'run_id':run.id,'test_case_id':case.id,'source':'live','turns':observed,'updated_at':now()})
                            evidence=Evidence.model_validate(observed[-1]); evidence.turns=observed
                            evidence.latency_ms=(time.monotonic()-started)*1000
                            evidence.usage={key:sum(t['usage'].get(key,0) for t in observed) for key in ('input_tokens','output_tokens','total_tokens') if all(isinstance(t['usage'].get(key),(int,float)) for t in observed)}
                            if all(isinstance(t['usage'].get('estimated_cost'),(int,float)) for t in observed):evidence.usage['estimated_cost']=sum(t['usage']['estimated_cost'] for t in observed)
                            if all(isinstance(t['usage'].get('model_calls'),int) for t in observed):evidence.usage['model_calls']=sum(t['usage']['model_calls'] for t in observed)
                            if all(isinstance(t['usage'].get('cost'),(int,float)) and t['usage']['cost']>=0 for t in observed):evidence.usage['cost']=sum(t['usage']['cost'] for t in observed)
                        findings=evaluate(case,evidence); judges=[]
                        if plan.judge:
                            from .llm import judge
                            async with budget_lock:
                                if eval_spent+RESERVATION>plan.evaluation_budget+1e-9: raise ValueError('Judge budget exhausted')
                                eval_spent+=RESERVATION
                            judges=await judge(case,evidence,self.store,actor)
                        verdict=aggregate([f.verdict for f in findings]+[j.verdict for j in judges])
                        result=Result(test_case_id=case.id,verdict=verdict,evidence=evidence,findings=findings,judge_findings=judges)
                except asyncio.CancelledError: raise
                except LookupError:
                    result=Result(test_case_id=case.id,verdict=Verdict.INCONCLUSIVE,evidence=evidence,findings=[Finding(criterion='historical_evidence',verdict=Verdict.INCONCLUSIVE,explanation='No matching imported historical evidence selected')])
                except Exception as exc:
                    error_category='timeout' if isinstance(exc,asyncio.TimeoutError) else 'budget' if 'budget' in str(exc).lower() else 'infrastructure'
                    evidence.error={'category':error_category,'exception_type':type(exc).__name__}
                    result=Result(test_case_id=case.id,verdict=Verdict.ERROR,evidence=evidence,error_category=error_category,findings=[Finding(criterion='execution',verdict=Verdict.ERROR,explanation='Execution failed: '+error_category+'; inspect server logs without exposing credentials')])
                finally:
                    if session:
                        try: await asyncio.wait_for(adapter.cleanup(session),5)
                        except Exception: run.reproducibility['cleanup_error']=True
                run.results.append(result)
                reported=result.evidence.usage.get('cost')
                if result.evidence.source=='live':
                    if isinstance(reported,(float,int)) and reported>=0 and not cost_unknown: run.measured_cost=(run.measured_cost or 0)+reported
                    else: cost_unknown=True; run.measured_cost=None
                self.store.put('run',run)
        workers=[]
        try:
            workers=[asyncio.create_task(one(c)) for c in plan.snapshots]
            await asyncio.gather(*workers)
            run.status='completed'
        except asyncio.CancelledError:
            for task in workers: task.cancel()
            await asyncio.gather(*workers,return_exceptions=True)
            run.status='cancelled'; run.cancellation_requested=True
        except Exception:
            run.status='error'
        finally:
            run.ended_at=now(); run.reproducibility['evaluation_reserved_cost']=eval_spent
            self.store.put('run',run); self.store.audit(actor,'run_finished',{'run_id':run.id,'status':run.status,'results':len(run.results)})
            self.tasks.pop(run.id,None)
