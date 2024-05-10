from pydantic import BaseModel
from typing import List, Dict, Optional

class reference(BaseModel):
    url: Optional[str] = ""
    title: Optional[str] = ""
    score: Optional[float] = ""

class queryLLMElserResponse(BaseModel):
    llm_response: str
    references: List[reference]
    model_id: str
    query_type: str
    error: Optional[str] = ""
