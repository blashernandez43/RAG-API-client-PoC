import os
import uvicorn
import sys

from dotenv import load_dotenv

# Fast API
from fastapi import FastAPI, Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN, HTTP_500_INTERNAL_SERVER_ERROR
from fastapi.middleware.cors import CORSMiddleware

# ElasticSearch
from elasticsearch import AsyncElasticsearch

# wx.ai
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


# Custom type classes
from customTypes.queryLLMElserResponse import queryLLMElserResponse
from customTypes.queryLLMElserRequest import queryLLMElserRequest, LLMParams



# wx.ai
from ibm_watson_machine_learning.foundation_models import Model

app = FastAPI()

# Set up CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
# RAG APP Security
API_KEY_NAME = "RAG-APP-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

#Token to IBM Cloud
ibm_cloud_api_key = os.environ.get("IBM_CLOUD_API_KEY")
project_id = os.environ.get("WX_PROJECT_ID")

# wxd creds
wxd_creds = {
    "username": os.environ.get("WXD_USERNAME"),
    "password": os.environ.get("WXD_PASSWORD"),
    "wxdurl": os.environ.get("WXD_URL")
}

wd_creds = {
    "apikey": os.environ.get("WD_API_KEY"),
    "wd_url": os.environ.get("WD_URL")
}

# WML Creds
wml_credentials = {
    "url": os.environ.get("WX_URL"),
    "apikey": os.environ.get("IBM_CLOUD_API_KEY")
}

# Create a global client connection to elastic search
async_es_client = AsyncElasticsearch(
    wxd_creds["wxdurl"],
    basic_auth=(wxd_creds["username"], wxd_creds["password"]),
    verify_certs=True,
    request_timeout=7200,
)


model_id = os.environ.get("LLM_MODEL_ID")
decoding_method = os.environ.get("DECODING_METHOD")
max_tokens = int(os.environ.get("MAX_TOKENS"))
min_tokens = int(os.environ.get("MIN_TOKENS"))



llm_params = LLMParams(model_id=model_id, parameters={"decoding_method": decoding_method, "max_new_tokens": max_tokens, "min_new_tokens": min_tokens})

llm_instructions = os.environ.get("LLM_INSTRUCTIONS")

model = Model(
model_id=model_id,
params=llm_params.parameters.dict(),
credentials=wml_credentials,
project_id=project_id
)



# Basic security for accessing the App
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == os.environ.get("RAG_APP_API_KEY"):
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate RAG APP credentials. Please check your ENV."
        )

@app.get("/")
def index(api_key: str = Security(get_api_key)):
    return {"Hello": "World"}


@app.post("/queryWXDLLM")
async def queryWXDLLM(request: queryLLMElserRequest, api_key: str = Security(get_api_key))->queryLLMElserResponse:
    question         = request.question
    num_results      = request.num_results
    
    index_name       = "search-mist-documentation-webcrawl"
    es_model_name    = ".elser_model_2_linux-x86_64"
    min_confidence = 10
    
    # Sanity check for instructions
    if "{query_str}" not in llm_instructions or "{context_str}" not in llm_instructions:
        data_response = {
            "llm_response": "",
            "references": [],
            "error": "LLM instructions must contain {query_str} and {context_str}"
        }
        return queryLLMElserResponse(**data_response)
    
    # Query indexes
    try:
        query_nested_index = await async_es_client.search(
                index=index_name,
                query={
                        "nested": {
                            "path": "passages",
                            "query": {
                            "text_expansion": {
                                "passages.sparse.tokens": {
                                "model_id": es_model_name,
                                "model_text": question
                                }
                            }
                            },
                            "inner_hits": {"_source": {"excludes": ["passages.sparse"]}}
                        },
                        "boosting": {
                            "positive": {
                                "term": {
                                    "text": question
                                }
                            },
                        },
                    },
                size=num_results,
                min_score=min_confidence                
        )
    except Exception as e:
        return {"msg": "Error searching index", "error": e}
    
    # Get relevant chunks and format
    #relevant_chunks = [query_regular_index, query_nested_index]
    
    #hits_index1 = [hit for hit in relevant_chunks[0]["hits"]["hits"]] #support portal
    hits_index2 = [hit for hit in query_nested_index["hits"]["hits"]]
    context2_preprocess = []
    for hit in hits_index2:
        for passage in hit["_source"]["passages"]:
            context2_preprocess.append(passage["text"])
    
    
    #context1 = "\n\n\n".join([rel_ctx["_source"]['Text'] for rel_ctx in hits_index1])
    #context1 = "\n" #removing support portal query
    context = "\n\n\n.".join(context2_preprocess)
    prompt_text = get_custom_prompt(llm_instructions, context, question)
    print("\n\n\n", prompt_text)    
    
    # LLM answer generation
    print(model.params.items())
    model_res = model.generate_text(prompt_text)
    
    # LLM references formatting
    
    uniform_format = {
        "url": ["url"],
        "title": ["title"],
        "score": ["score"],
        #"text": ["Text", "text"]
    }
    
    #references_context1 = [(chunks["_source"], chunks["_score"]) for chunks in relevant_chunks[0]["hits"]["hits"]]
    references_context2 = [(chunks["_source"], chunks["_score"]) for chunks in query_nested_index["hits"]["hits"]]
    
    references = []
    
    #for (ref, score) in references_context1:
    #    ref["score"] = score
        #references.append(convert_to_uniform_format(ref, uniform_format)) Hiding support portal references
    
    for (ref, score) in references_context2:
        for passage in ref["passages"]:
            passage["score"] = score
            references.append(convert_to_uniform_format(passage, uniform_format))
            
    references = sort_and_delete_duplicates(references, sort_key="score", unique_key="url")
    
    res = {
        "llm_response": model_res,
        "references": references
        
    }
    
    return queryLLMElserResponse(**res)


def get_custom_prompt(llm_instructions, wd_context, query_str):
    #context_str = "\n".join(wd_contexts)

    # Replace the placeholders in llm_instructions with the actual query and context
    prompt = llm_instructions.replace("{query_str}", query_str).replace("{context_str}", wd_context)
    return prompt

def convert_to_uniform_format(obj, uniform_format):
    uniform_obj = {}
    for key, possible_keys in uniform_format.items():
        for possible_key in possible_keys:
            if possible_key in obj:
                uniform_obj[key] = obj[possible_key]
                break
        if key not in uniform_obj:
            uniform_obj[key] = None
    return uniform_obj

def sort_and_delete_duplicates(obj_list, sort_key, unique_key):
    sorted_objects = sorted(obj_list, key=lambda x: x[sort_key], reverse=True)
    
    unique_objects = []
    seen_keys = set()
    for obj in sorted_objects:
        if obj[unique_key] not in seen_keys:
            unique_objects.append(obj)
            seen_keys.add(obj[unique_key])
            
    return unique_objects
    

if __name__ == '__main__':
    if 'uvicorn' not in sys.argv[0]:
        uvicorn.run("app:app", host='0.0.0.0', port=4050, reload=True)
