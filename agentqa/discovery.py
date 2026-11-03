"""Read-only AST discovery. Never import target modules or open target databases."""
import ast
import os
import hashlib
import json
from pathlib import Path
from .models import AgentProfile, Capability, Mode, Risk

REFERENCE=Path(os.getenv('AGENTQA_HR_REFERENCE') or str(Path(__file__).resolve().parent.parent/'private/hr-agent')).expanduser().resolve()

def fingerprint(profile):
    data=profile.model_dump(mode='json',by_alias=True,exclude={'fingerprint'})
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

def annotation_schema(node, aliases=None):
    aliases=aliases or {}
    if isinstance(node,ast.Constant) and node.value is None: return {'type':'null'}
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.BitOr):
        return {'anyOf':[annotation_schema(node.left,aliases),annotation_schema(node.right,aliases)]}
    if isinstance(node,ast.Name):
        if node.id in aliases: return dict(aliases[node.id])
        return {'type': {'str':'string','int':'integer','float':'number','bool':'boolean','dict':'object','list':'array'}.get(node.id,'string')}
    if isinstance(node,ast.Subscript):
        name=ast.unparse(node.value)
        pieces=node.slice.elts if isinstance(node.slice,ast.Tuple) else [node.slice]
        if name in ('Optional','Union'):
            branches=[annotation_schema(piece,aliases) for piece in pieces]
            if name=='Optional':branches.append({'type':'null'})
            return {'anyOf':branches}
        if name=='Literal': return {'enum':[ast.literal_eval(n) for n in pieces]}
        if name=='Annotated':
            s=annotation_schema(pieces[0],aliases)
            for p in pieces[1:]:
                if isinstance(p,ast.Call) and ast.unparse(p.func)=='Field':
                    for k in p.keywords:
                        mapped={'ge':'minimum','le':'maximum','gt':'exclusiveMinimum','lt':'exclusiveMaximum','min_length':'minLength','max_length':'maxLength','pattern':'pattern'}.get(k.arg)
                        if mapped:
                            if s.get('type')=='array':mapped={'minLength':'minItems','maxLength':'maxItems'}.get(mapped,mapped)
                            try: s[mapped]=ast.literal_eval(k.value)
                            except (ValueError,TypeError): pass
            return s
        if name=='list': return {'type':'array','items':annotation_schema(pieces[0],aliases)}
    return {'description':ast.unparse(node) if node else 'unknown'}

def nullable(schema):
    schema=dict(schema)
    if 'type' in schema:
        types=schema['type'] if isinstance(schema['type'],list) else [schema['type']]
        schema['type']=list(dict.fromkeys(types+['null']))
    elif 'anyOf' in schema:
        if not any(x.get('type')=='null' for x in schema['anyOf']):schema['anyOf']=[*schema['anyOf'],{'type':'null'}]
    else:schema={'anyOf':[schema,{'type':'null'}]}
    if 'enum' in schema and None not in schema['enum']:schema['enum']=[*schema['enum'],None]
    return schema


def field_description_overrides(tree):
    """Read literal operation/field branches only; never execute target code."""
    overrides={}
    for fn in tree.body:
        if not isinstance(fn,ast.FunctionDef) or fn.name!='build_schema':continue
        for node in ast.walk(fn):
            if not isinstance(node,ast.If) or not isinstance(node.test,ast.BoolOp) or not isinstance(node.test.op,ast.And):continue
            conditions={}
            for test in node.test.values:
                if isinstance(test,ast.Compare) and len(test.ops)==1 and isinstance(test.ops[0],ast.Eq) and isinstance(test.comparators[0],ast.Constant):
                    conditions[ast.unparse(test.left)]=test.comparators[0].value
            if 'fn.__name__' not in conditions or 'name' not in conditions:continue
            for assignment in node.body:
                if isinstance(assignment,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='description' for t in assignment.targets) and isinstance(assignment.value,ast.Constant):
                    overrides[(conditions['fn.__name__'],conditions['name'])]=(assignment.value.value,node.lineno)
    return overrides


def source_profile(path=REFERENCE):
    path=Path(path).resolve()
    # Fixed reference root only: arbitrary filesystem traversal is not an API feature.
    if path!=REFERENCE.resolve(): raise ValueError('Source discovery is restricted to the configured read-only HR reference')
    schema_tree=ast.parse((path/'operation_schemas.py').read_text())
    aliases={}
    overrides=field_description_overrides(schema_tree)
    for n in schema_tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            if isinstance(n.value,ast.Subscript): aliases[n.targets[0].id]=annotation_schema(n.value,aliases)
    auth_rules={}
    for node in schema_tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_authorize':
            rule=ast.literal_eval(node.value.args[0]); names=ast.literal_eval(node.value.args[1])
            for name in names.split():auth_rules[name]=rule
    fields={}
    for n in schema_tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIELDS' for t in n.targets):
            for k,v in zip(n.value.keys,n.value.values):
                s=annotation_schema(v.elts[0],aliases)
                s['description']=ast.literal_eval(v.elts[1])
                fields[ast.literal_eval(k)]=s
    caps=[]; evidence=['operation_schemas.py:sha256:'+hashlib.sha256((path/'operation_schemas.py').read_bytes()).hexdigest()]
    for file in sorted((path/'services').glob('*.py')):
        content=file.read_text(); digest=hashlib.sha256(content.encode()).hexdigest()
        evidence.append(f'{file.name}:sha256:{digest}')
        tree=ast.parse(content)
        for fn in tree.body:
            if not isinstance(fn,ast.FunctionDef): continue
            deco=next((d for d in fn.decorator_list if isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='operation'),None)
            if not deco: continue
            module=ast.literal_eval(deco.args[0]); write=any(k.arg=='write' and ast.literal_eval(k.value) for k in deco.keywords)
            defaults=[None]*(len(fn.args.args)-len(fn.args.defaults))+fn.args.defaults
            props={}; required=[]
            for arg,default in zip(fn.args.args,defaults):
                props[arg.arg]=dict(fields.get(arg.arg,annotation_schema(arg.annotation,aliases)))
                if default is None: required.append(arg.arg)
                else:
                    try:
                        value=ast.literal_eval(default)
                        if value is None:props[arg.arg]=nullable(props[arg.arg])
                        props[arg.arg]['default']=value
                    except (ValueError,TypeError): pass
                if (fn.name,arg.arg) in overrides:
                    description,line=overrides[(fn.name,arg.arg)]
                    props[arg.arg]['description']=description
                    props[arg.arg]['x-provenance']=f'operation_schemas.py:{line}:literal operation-specific description'
            caps.append(Capability(name=fn.name,tool='get_employee' if fn.name=='get_employee' else module+'_operations',description=ast.get_docstring(fn) or '',schema={'type':'object','properties':props,'required':required,'additionalProperties':False},risk=Risk.PROPOSE if write else Risk.READ,provenance='verified',evidence=[f'{file.name}:{fn.lineno}:static AST; runtime unverified'],authorization=auth_rules.get(fn.name,'Unknown; backend enforcement not runtime verified')))
    profile=AgentProfile(framework='LangChain create_agent / LangGraph (source)',capabilities=caps,tools=sorted({c.tool for c in caps}|{'describe_operation'}),model_metadata={'declared_default':'gpt-4.1-mini','provenance':'Source default only; runtime environment overrides unverified','underlying_supported_modes':[m.value for m in Mode]},conversation=True,observable_events=['HTTP final response; Studio messages when configured'],supported_modes=[Mode.GENERATE,Mode.STATIC,Mode.HISTORICAL,Mode.MOCK],unknowns=['Runtime authorization and tool traces unverified','Source constraints include custom validators not fully representable in JSON Schema','Live endpoint and isolated environment not verified','HTTP /chat is stateless; Studio memory is configuration-dependent'],evidence=evidence)
    profile.fingerprint=fingerprint(profile)
    return profile
