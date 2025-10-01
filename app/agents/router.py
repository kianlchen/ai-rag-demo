from fastapi import APIRouter
from .agent import Agent
from .tools import REGISTRY
from .schemas import AgentRequest, AgentResponse

router = APIRouter(prefix="/agent", tags=["agent"])
_agent = Agent(REGISTRY)

@router.post("", response_model=AgentResponse)
def run_agent(req: AgentRequest):
    return _agent.run(req.text)