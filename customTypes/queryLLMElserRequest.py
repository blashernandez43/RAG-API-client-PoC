from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class Moderations(BaseModel):
    hap_input: str = 'true'
    threshold: float = 0.75
    hap_output: str = 'true'

class Parameters(BaseModel):
    decoding_method: str = "greedy"
    min_new_tokens: int = 1
    max_new_tokens: int = 500
    moderations: Moderations = Moderations()

    def dict(self, *args, **kwargs):
        """
        Override dict() method to return a dictionary representation
        """
        params_dict = super().dict(*args, **kwargs)
        params_dict['moderations'] = self.moderations.dict()
        return params_dict

class LLMParams(BaseModel):
    model_id: str = 'ibm/granite-13b-chat-v2'
    inputs: list = []
    parameters: Parameters = Parameters()

    # Resolves warning error with model_id:
    #     Field "model_id" has conflict with protected namespace "model_".
    #     You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
    #     warnings.warn(
    class Config:
        protected_namespaces = ()

class queryLLMElserRequest(BaseModel):
    question: str
    num_results: Optional[str] = Field(default="5")
    llm_params: Optional[LLMParams] = LLMParams()

