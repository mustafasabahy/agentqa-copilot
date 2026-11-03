"""Second agent: fictional warehouse, independent REST interface and no paid models."""
from fastapi import FastAPI
from pydantic import BaseModel
from agentqa.adapters import DemoAdapter
app=FastAPI(title='Fictional Warehouse REST Agent')
adapter=DemoAdapter()
class Request(BaseModel):
    message:str
    session_id:str
@app.get('/capabilities')
async def capabilities():return (await adapter.discover()).model_dump(mode='json',by_alias=True)
@app.post('/chat')
async def chat(request:Request):
    adapter.sessions.setdefault(request.session_id,{})
    e=await adapter.invoke(request.message,request.session_id)
    return {'response':e.final,'tool_calls':e.tool_calls,'usage':{'cost':0}}

@app.get('/.well-known/agentqa-isolation')
def isolation():return {'deployment_id':'fictional-warehouse-local-v1','environment':'isolated','fictional_data':True}
