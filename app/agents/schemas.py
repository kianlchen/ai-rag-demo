from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    text: str = Field(..., min_length=1)

class AgentResult(BaseModel):
    tool: str
    output: str

class AgentResponse(BaseModel):
    tool: str
    output: str
    history: list[AgentResult]