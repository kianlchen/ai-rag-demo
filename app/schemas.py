from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Summary of request")
    max_words: int = Field(80, ge=10, le=400)


class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="Summary of response")
    confidence: float = Field(..., ge=0.0, le=1.0)
