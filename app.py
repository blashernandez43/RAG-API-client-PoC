import json
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
from customTypes.queryLLMElserRequest import queryLLMElserRequest


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
    request_timeout=3600,
)

# Create a watsonx client cache for faster calls.
custom_watsonx_cache = {}

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
    #llm_params       = request.llm_params
    index_names       = [
    "juniper-knowledgebase-api-v2",
    "search-juniper-documentation-chunked"
  ]
    #llm_instructions = request.llm_instructions
    es_model_name    = ".elser_model_2_linux-x86_64"
    min_confidence = 10


    # Sets the llm params if the user provides it
    if not llm_params:
        llm_params = {
          "parameters": {
              "decoding_method": "greedy",
              "max_new_tokens": 500,
              "min_new_tokens": 0,
              "stop_sequences": [],
              "repetition_penalty": 1
            },
        "model_id": "mistralai/mixtral-8x7b-instruct-v01", 
    }

    # Sets the llm instruction if the user provides it
    if not llm_instructions:
        llm_instructions = "[INST]<<SYS>>You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Be brief in your answers. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'\''t know the answer to a question, please do not share false information. <</SYS>>\nGenerate the next agent response by answering the question. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities and use the tiles of documents to separate between topics or domains. Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer.\n{context_str}<</SYS>>\n\n{query_str}. Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer. [/INST]"
    
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
        relevant_chunks = []
        query_regular_index = await async_es_client.search(
            index=index_names[0],
            query={
            "text_expansion": {
                "tokens": {
                    "model_id": es_model_name,
                    "model_text": question,
                    }
                }
            },
            size=num_results,
            min_score=min_confidence
        )
        query_nested_index = await async_es_client.search(
                index=index_names[1],
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
                        }
                    },
                size=num_results,
                min_score=min_confidence                
        )
    except Exception as e:
        return {"msg": "Error searching indexes", "error": e}
    
    # Get relevant chunks and format
    relevant_chunks = [query_regular_index, query_nested_index]
    
    hits_index1 = [hit for hit in relevant_chunks[0]["hits"]["hits"]]
    hits_index2 = [hit for hit in relevant_chunks[1]["hits"]["hits"]]
    context2_preprocess = []
    for hit in hits_index2:
        for passage in hit["_source"]["passages"]:
            context2_preprocess.append(passage["text"])
    
    
    context1 = "\n\n\n".join([rel_ctx["_source"]['Text'] for rel_ctx in hits_index1])
    context2 = "\n\n\n.".join(context2_preprocess)
    prompt_text = get_custom_prompt(llm_instructions, [context1, context2], question)
    print("\n\n\n\n", prompt_text)
    
    
    model = get_custom_watsonx(llm_params.model_id, llm_params.parameters.dict())
    # LLM answer generation
    model_res = model.generate_text(prompt_text)
    
    # LLM references formatting
    
    uniform_format = {
        "url": ["url"],
        "title": ["title"],
        "score": ["score"],
        #"text": ["Text", "text"]
    }
    
    references_context1 = [(chunks["_source"], chunks["_score"]) for chunks in relevant_chunks[0]["hits"]["hits"]]
    references_context2 = [(chunks["_source"], chunks["_score"]) for chunks in relevant_chunks[1]["hits"]["hits"]]
    
    references = []
    
    for (ref, score) in references_context1:
        ref["score"] = score
        references.append(convert_to_uniform_format(ref, uniform_format))
    
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
 

@app.post("/testConfidence")
async def testConfidence(index: int, question: str, num_results: int, min_score: float):
    index_names       = [
    "juniper-knowledgebase-api-v2",
    "search-juniper-documentation-chunked"
  ]
    es_model_name    = ".elser_model_2_linux-x86_64"
    
    query_regular_index = await async_es_client.search(
            index=index_names[0],
            query={
            "text_expansion": {
                "tokens": {
                    "model_id": es_model_name,
                    "model_text": question,
                    }
                }
            },
            size=num_results,
            min_score=min_score
        )
    
    query_nested_index = await async_es_client.search(
                index=index_names[1],
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
                        }
                    },
                size=num_results,
                min_score=min_score                       
        )
    queries = [query_regular_index["hits"]["hits"], query_nested_index["hits"]["hits"]]
    
    return queries[index]


def get_custom_watsonx(model_id, additional_kwargs):
    # Serialize additional_kwargs to a JSON string, with sorted keys
    additional_kwargs_str = json.dumps(additional_kwargs, sort_keys=True)
    # Generate a hash of the serialized string
    additional_kwargs_hash = hash(additional_kwargs_str)
    
    cache_key = f"{model_id}_{additional_kwargs_hash}"

    # Check if the object already exists in the cache
    if cache_key in custom_watsonx_cache:
        print("I had model in memory")
        return custom_watsonx_cache[cache_key]
        
    model = Model(
    model_id=model_id,
    params=additional_kwargs,
    credentials=wml_credentials,
    project_id=project_id
    )
    custom_watsonx_cache[cache_key] = model
    return model

def get_custom_prompt(llm_instructions, wd_contexts, query_str):#
    context_str = "\n".join(wd_contexts)

    # Replace the placeholders in llm_instructions with the actual query and context
    prompt = llm_instructions.replace("{query_str}", query_str).replace("{context_str}", context_str)
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
