from pydantic import BaseModel
from typing import List, Dict, Optional

class reference(BaseModel):
    url: Optional[str]
    title: Optional[str]
    score: Optional[float]
    error: Optional[str]

class queryLLMElserResponse(BaseModel):
    llm_response: str
    references: List[reference]
