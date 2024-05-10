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
from elasticsearch import AsyncElasticsearch, Elasticsearch

# wx.ai
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


# Custom type classes
from customTypes.queryLLMElserResponse import queryLLMElserResponse
from customTypes.queryLLMElserRequest import queryLLMElserRequest, LLMParams
from JuniperMultiModel import JuniperMultiModel
import asyncio
import aiohttp

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

# ELSER Model elastic connection

es_client_elser = Elasticsearch(
    os.environ.get("ELSER_ES_URL"),
    basic_auth=(os.environ.get("ELSER_ES_USER"),os.environ.get("ELSER_ES_PASS")),
    verify_certs=True,
    request_timeout=10000
)
# Dense Model elastic connection

es_client_dense = Elasticsearch(
    os.environ.get("DENSE_ES_URL"),
    basic_auth=(os.environ.get("DENSE_ES_USER"),os.environ.get("DENSE_ES_PASS")),
    verify_certs=True,
    request_timeout=10000
)
"""
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
"""
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

"""
@app.post("/queryWXDLLM")
async def queryWXDLLM(request: queryLLMElserRequest, api_key: str = Security(get_api_key))->queryLLMElserResponse:
    question         = request.question
    num_results      = request.num_results
    
    index_names       = [
    "juniper_knowledge_base_dest",
    "search-mist-documentation-webcrawl"
  ]
    es_model_name    = ".elser_model_2_linux-x86_64"
    min_confidence = 9.5
    
    # Sanity check for instructions
    if "{query_str}" not in llm_instructions or "{context_str}" not in llm_instructions:
        data_response = {
            "llm_response": "",
            "references": [],
            "error": "LLM instructions must contain {query_str} and {context_str}"
        }
        return queryLLMElserResponse(**data_response)
    #Hardcode number of responses
    num_results = 4

    # Query indexes
    try:
        relevant_chunks = []
        query_regular_index = await async_es_client.search(
            index=index_names[0],
            query={
            "text_expansion": {
                "content_embedding": {
                    "model_id": es_model_name,
                    "model_text": question,
                    }
                }
            },
            #size=num_results,
            size=3,
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
                size=5,
                min_score=min_confidence                
        )
    except Exception as e:
        return {"msg": "Error searching indexes", "error": e}
    
    # Get relevant chunks and format
    relevant_chunks = [query_regular_index, query_nested_index]
    
    hits_index1 = [hit for hit in relevant_chunks[0]["hits"]["hits"]]
    hits_index2 = [hit for hit in relevant_chunks[1]["hits"]["hits"]]
    #hits = (text, relevance)
    #print("Hits indices:")
    #print(hits_index1, hits_index2)
    context2_preprocess = []
    for hit in hits_index2:
        for passage in hit["_source"]["passages"]:
            print("2. Appending text of length " + str(len(passage["text"])))
            context2_preprocess.append(passage["text"])
    context2 = context_str = ("\n".join(context2_preprocess))[:80000] #Limit to 50k chars
    print(context2)
    for rel in hits_index1:
        print("1. Appending text of length " + str(len(rel["_source"]["text"])))

    context1 = "\n\n".join([rel_ctx["_source"]['text'] for rel_ctx in hits_index1])
    context2 = "\n\n"+context2
    prompt_text = get_custom_prompt(llm_instructions, [context1, context2], question)
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
    references = references[0:3]

    res = {
        "llm_response": model_res,
        "references": references
    }
    
    return queryLLMElserResponse(**res)
"""

## end to end call retrieve document from elasticsearch and with the context make LLM call to watsonx.ai
@app.post("/queryWXDLLM")
def pipeline(request: queryLLMElserRequest, api_key: str = Security(get_api_key)) -> list[queryLLMElserResponse]:
    question         = request.question
    juniper = JuniperMultiModel()
    elserContext, denseContext, elser_metadata, dense_metadata = juniper.retrieve_documents(
        question=question,
        es_elser_client=es_client_elser, 
        es_dense_client=es_client_dense)

    mixtral_elser = juniper.build_prompt(context=elserContext,model_id="MIXTRAL", user_query=question)
    #lama3_elser = juniper.build_prompt(context=elserContext,model_id="LLAMA3", user_query=question)
    mixtral_dense = juniper.build_prompt(context=denseContext,model_id="MIXTRAL", user_query=question)
    #lama3_dense = juniper.build_prompt(context=denseContext,model_id="LLAMA3", user_query=question)

    async def fetch_all_results():
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                juniper.send_to_watsonxai(prompts=mixtral_elser, model_id="MIXTRAL", type="elser",wml_credentials=wml_credentials, metadata=elser_metadata),
                #juniper.send_to_watsonxai(prompts=lama3_elser, model_id="LLAMA3", type="elser",wml_credentials=wml_credentials),
                juniper.send_to_watsonxai(prompts=mixtral_dense, model_id="MIXTRAL", type="dense",wml_credentials=wml_credentials, metadata=dense_metadata),
                #juniper.send_to_watsonxai(prompts=lama3_dense, model_id="LLAMA3", type="dense",wml_credentials=wml_credentials),
            )
            return results

    all_results = asyncio.run(fetch_all_results())
    return all_results

def get_custom_prompt(llm_instructions, wd_contexts, query_str):#
    context_str = "\n\n".join(wd_contexts)
    
    # Replace the placeholders in llm_instructions with the actual query and context
    prompt = llm_instructions.replace("{query_str}", query_str).replace("{context_str}", context_str) + "\n"
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