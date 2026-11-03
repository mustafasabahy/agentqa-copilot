"""Model intent -> deterministic contract-grounded drafts. No target invocation."""
import json
from typing import Literal
from pydantic import Field,StrictInt,StrictStr,JsonValue
import jsonschema
from .models import Contract,TestCase,Assertion,Suite,uid,now
from .capabilities import execution_capabilities
from .generation import validate_case,deduplicate

class ContractExpectedValue(Contract):
    value:JsonValue
    schema_:dict=Field(alias="schema")
    def checked(self):
        jsonschema.validate(self.value,self.schema_);return self.value

class TestScenarioProposal(Contract):
    operation_id:StrictStr
    objective:StrictStr=Field(min_length=1)
    user_inputs:list[StrictStr]=Field(min_length=1,max_length=10)
    arguments:dict=Field(default_factory=dict)
    intent:Literal['valid_arguments','missing_required','invalid_arguments']='valid_arguments'
    missing_argument:StrictStr|None=None
    rationale:StrictStr=''

class GenerationRequest(Contract):
    agent_id:StrictStr
    count:StrictInt=Field(ge=1,le=40)
    focus:StrictStr=''
    constraints:list[StrictStr]=Field(default_factory=list)

def proposal_schema(agent):
    schema=TestScenarioProposal.model_json_schema()
    schema['properties']['operation_id']['enum']=[c.name for c in agent.profile.capabilities]
    return {'name':'ScenarioBatch','description':'Propose intent only; executable fields are server-owned','parameters':{'type':'object','properties':{'proposals':{'type':'array','maxItems':20,'items':schema}},'required':['proposals'],'additionalProperties':False}}

class CompilationError(ValueError):
    def __init__(self,field,message):self.field=field;super().__init__(message)

def compile_proposal(agent,raw,index=0):
    proposal=TestScenarioProposal.model_validate(raw)
    cap=next((c for c in agent.profile.capabilities if c.name==proposal.operation_id),None)
    if cap is None:raise CompilationError('operation_id','Unknown discovered operation')
    schema=cap.schema_;expected='Follow the discovered operation contract; backend results remain unknown until observed.'
    assertions=[]
    if proposal.intent=='missing_required':
        missing=proposal.missing_argument
        if missing not in schema.get('required',[]) or missing in proposal.arguments:raise CompilationError('missing_argument','Name an omitted required argument; optional/default fields do not require clarification')
        reduced={**schema,'required':[k for k in schema.get('required',[]) if k!=missing]}
        try:jsonschema.validate(proposal.arguments,reduced)
        except jsonschema.ValidationError as exc:raise CompilationError('arguments.'+'.'.join(map(str,exc.path)),exc.message) from None
        expected=f'Clarify missing required {missing} when no trusted context supplies it; do not invent identity or results.'
        assertions=[Assertion(kind='exists',path='final')]
    else:
        errors=list(jsonschema.Draft202012Validator(schema).iter_errors(proposal.arguments))
        if proposal.intent=='valid_arguments' and errors:raise CompilationError('arguments.'+'.'.join(map(str,errors[0].path)),errors[0].message)
        if proposal.intent=='invalid_arguments':
            if not errors:raise CompilationError('arguments','Invalid-argument scenario must actually violate the operation schema')
            expected='Explain a contract validation problem using observed evidence; do not fabricate a successful result.'
            assertions=[Assertion(kind='exists',path='final')]
        else:
            assertions=[Assertion(kind='tool_selection',path='tool_calls',expected=cap.tool)]
            if cap.tool.endswith('_operations'):
                assertions.append(Assertion(kind='operation_routing',path='tool_calls',expected=cap.name))
                path=f'tool_calls[operation={cap.name}].arguments.arguments'
            else:path=f'tool_calls[name={cap.tool}].arguments'
            # Supplied optional values must be preserved; omitted defaults may be materialized by the target.
            for key,value in proposal.arguments.items():assertions.append(Assertion(kind='equals',path=path+'.'+key,expected=ContractExpectedValue(value=value,schema=schema['properties'][key]).checked()))
    modes=execution_capabilities(agent)['currently_available_modes']
    case=TestCase(name=f'{cap.name}: {proposal.objective[:150]}',objective=proposal.objective,capability=cap.name,category=proposal.intent,turns=proposal.user_inputs,expected_behavior=expected,assertions=assertions,judge_rubric=['Assess observed contract compliance; INCONCLUSIVE when outcomes or required tool evidence are missing'],evidence_requirements=['final','tool_calls'],risk=cap.risk,modes=modes,preconditions=['Isolated reviewed fixtures; supplied arguments are proposed data, never authenticated identity']+(['No trusted context supplies the missing argument'] if proposal.intent=='missing_required' else []),reviewed=False,review_notes='Compiled from discovered contract. Model rationale is untrusted review context: '+proposal.rationale)
    validate_case(case,agent.profile)
    return case

async def generate(store,agent,actor,count,focus,constraints,proposer,source):
    req=GenerationRequest(agent_id=agent.id,count=count,focus=focus,constraints=constraints)
    suite=Suite(agent_id=agent.id,name=f'{agent.name} · compiled {count} scenario drafts',cases=[],generation=source)
    result={'id':uid(),'at':now(),'agent_id':agent.id,'suite_id':suite.id,'requested':count,'generated':0,'validated':0,'persisted':0,'reviewed':0,'rejected':[],'batches':[],'status':'in_progress','source':source}
    store.put('suite',suite);store.put('compiled_generation',result)
    try:
        for batch_index,offset in enumerate(range(0,count,20)):
            expected=min(20,count-offset)
            raw=await proposer(agent,expected,focus,constraints,batch_index)
            proposals=raw.get('proposals') if isinstance(raw,dict) else None
            if not isinstance(proposals,list):raise CompilationError('proposals','Expected a proposal array')
            if len(proposals)>expected:raise CompilationError('proposals',f'At most {expected} proposals allowed in this batch')
            accepted=[];result['generated']+=len(proposals)
            for i,proposal in enumerate(proposals):
                try:
                    case=compile_proposal(agent,proposal,i)
                    if len(deduplicate(suite.cases+accepted+[case]))!=len(suite.cases)+len(accepted)+1:raise CompilationError('user_inputs','Duplicate or near-duplicate proposal')
                    accepted.append(case)
                except Exception as exc:
                    if hasattr(exc,'errors'):issues=exc.errors(include_input=False,include_context=False)
                    else:issues=[{'field':getattr(exc,'field','proposal'),'message':str(exc)}]
                    result['rejected'].append({'batch':batch_index,'index':i,'issues':issues})
            suite.cases.extend(accepted);result['validated']+=len(accepted);result['persisted']=len(suite.cases)
            result['batches'].append({'index':batch_index,'expected':expected,'generated':len(proposals),'persisted':len(accepted)})
            result['status']='complete' if result['persisted']==count else 'partial'
            suite.generation=f"{source}; requested {count}; generated {result['generated']}; persisted {result['persisted']}; reviewed 0"
            with store.lock,store.connect() as c:
                c.execute('BEGIN IMMEDIATE')
                c.execute("UPDATE documents SET body=? WHERE kind='suite' AND id=?",(json.dumps(suite.model_dump(mode='json')),suite.id))
                c.execute("UPDATE documents SET body=? WHERE kind='compiled_generation' AND id=?",(json.dumps(result),result['id']))
            if len(proposals)!=expected or result['rejected']:break # no hidden refill or retry
    except Exception as exc:
        result['rejected'].append({'field':getattr(exc,'field','generation'),'error_type':type(exc).__name__,'message':str(exc)[:500]})
    result['status']='complete' if result['persisted']==count else 'partial' if result['persisted'] else 'failed'
    store.put('compiled_generation',result);store.audit(actor,'compiled_generation',result)
    return result
