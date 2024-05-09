import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util
import os,logging
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv
from transformers import AutoTokenizer
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn

#config Watsonx.ai environment
load_dotenv()
api_key = os.getenv("IBM_CLOUD_API_KEY", None)        
ibm_cloud_url = os.getenv("IBM_CLOUD_ENDPOINT", None) 
project_id = os.getenv("IBM_CLOUD_PROJECT_ID", None)

if api_key is None or ibm_cloud_url is None or project_id is None:
    raise Exception("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }


# sparse_index_name = "search-juniper-sparse-84k"
sparse_index_name = "search-juniper-sparse-99k-50overlap"
knn_index_name = "search-juniper-dense-minilm_l6_512-84k"
knapi_index="juniper-knowledgebase-api-v2"
jdc_index="search-juniper-documentation-chunked"

# es_endpoint= os.getenv('ES_END_POINT')
# es_user=os.getenv('ES_USER')
# es_password=os.getenv('ES_PASSWORD')

try:
    es = Elasticsearch(
    "https://juniper-index-test-alpha.es.us-west1.gcp.cloud.es.io",
    basic_auth=("elastic", "zX2tnZBrF3IaAXAvelfYzB9b"),
    verify_certs=True,
    request_timeout=10000
    )

    dense_es = Elasticsearch(
    "https://juniper-index-test-beta.es.us-central1.gcp.cloud.es.io",
    basic_auth=("elastic", "PKPB5pPfprKn3yWaplNxqJzJ"),
    verify_certs=True,
    request_timeout=10000
    )

except ConnectionError as e:
    print("Connection Error:", e)
    
if es.ping():
    print("Succesfully connected to ElasticSearch!!")
else:
    print("Oops!! Can not connect to Elasticsearch!")




def build_prompt(user_query,context):

    SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    USER_PROMPT = """A user has asked the following question: '{question}'. Use the information available from the '{context}' to provide a detailed and accurate answer in less than 150 words. If necessary, cite relevant sources or data to support your response. Ensure your answer is clearly structured and directly addresses the question.
    If the answer to the question is not available in the any of the provided context, please say: "I don't know the answer to that question." 
    While you have to answer only from the context, the answer language must be in the way such that it is coming from your own knowledge,

    Keep following things in mind:
    - Give the output with proper markdown styling features [#, -, * , ``, ```code `] .
    - Provide source attribution with Superscript within the answer and links at the end.
    """
    LLAMA3_PROMPT= """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Answer Based on the Provided Context: 
    """
    user_prompt = USER_PROMPT.format(question=user_query,context=context)
    formatted_prompt = LLAMA3_PROMPT.format(system_prompt=SYSTEM_PROMPT,user_prompt=user_prompt)
    return formatted_prompt


def send_to_watsonxai(prompts,
                    model_name="meta-llama/llama-3-70b-instruct",
                    decoding_method="greedy",
                    max_new_tokens=500,
                    min_new_tokens=30,
                    temperature=1.0,
                    repetition_penalty=1.0
                    ):

    assert not any(map(lambda prompt: len(prompt) < 1, prompts)), "make sure none of the prompts in the inputs prompts are empty"

    # Instantiate parameters for text generation
    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.REPETITION_PENALTY: repetition_penalty,
    }


    # Instantiate a model proxy object to send your requests
    model = Model(
        model_id=model_name,
        params=model_params,
        credentials=creds,
        project_id=project_id)

    response=model.generate_text(prompts)
    # for prompt in prompts:
    #     print(model.generate_text(prompt))
    return response

# Function to compute cosine similarity
def compute_cosine_similarity(input_text, response_text):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform([input_text, response_text])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return cosine_sim[0][0]


# Function to compute embeddings
def compute_sentence_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings

# Function to compute similarity using cosine similarity
def compute_similarity(embedding1, embedding2):
    # Compute cosine similarity between embeddings
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2)

    return similarity_score.item()

def search_knn(question):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector_of_input_keyword = model.encode(question)

    # query = {
    #     "field": "DescriptionVector",
    #     "query_vector": vector_of_input_keyword,
    #     "k": 10,
    #     "num_candidates": 500
    # }
    
    query =   {    "query": {
                    "multi_match" : {
                        "query":question,
                            "type":"best_fields",
                            "fields":[ "body_content", "title"]

                        }
                    },
                    "knn": {
                        "field": "content_embedding",
                        "query_vector": vector_of_input_keyword,
                        "k": 10,
                        "num_candidates": 100,
                        "boost": 10
                    },
                    #"size": 5
            }
    
    response = dense_es.search(
                index=knn_index_name,
                body=query,
                scroll='5m',  # Set the scroll timeout (e.g., 5 minutes)
                size=1  # Set the number of documents to retrieve per scroll
                )
    all_hits = response['hits']['hits']
    print(len(all_hits))

    return all_hits
def search_es(question):
    # question = "Whats in front panel of a JA2500?"
    # vector_of_input_keyword = model.encode(input_keyword)
    es_model_name    = ".elser_model_2_linux-x86_64"
    min_confidence = 10
    num_results =5
    top_k=20
    input_query= {
            "query": {
                "bool": { 
                "should": [
                    {
                    "text_expansion": {
                        "ml.tokens": {
                        "model_text":question,
                        "model_id": es_model_name
                        }
                    }
                    }
                ],
                "must": {
                    "multi_match" : {
                    "query":question,
                    "type":"best_fields",
                    "fields":[ "title", "body_content"]
                }
                }
                }
                },
            "min_score": 1 
            }
    
    response = es.search(
            index=sparse_index_name,
            # size=top_k,
            body=input_query,
            scroll='5m',  # Set the scroll timeout (e.g., 5 minutes)
            size=top_k  # Set the number of documents to retrieve per scroll
            )
    all_hits = response['hits']['hits']
    print(len(all_hits))
    flag = False

    return all_hits


def search_es_jdc(question):
    # question = "Whats in front panel of a JA2500?"
    # vector_of_input_keyword = model.encode(input_keyword)
    es_model_name    = ".elser_model_2_linux-x86_64"
    min_confidence = 10
    num_results =5

    input_query= {
                    "query": {
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
                    }
                    }
    response = es.search(
            index=jdc_index,
            body=input_query,
            scroll='5m',  # Set the scroll timeout (e.g., 5 minutes)
            size=3  # Set the number of documents to retrieve per scroll
            )
    all_hits = response['hits']['hits']
    print(len(all_hits))
    flag = False

    return all_hits


# knapi_index="juniper-knowledgebase-api-v2"
# jdc_index="search-juniper-documentation-chunked"
def search_es_knapi(question):
    # question = "Whats in front panel of a JA2500?"
    # vector_of_input_keyword = model.encode(input_keyword)
    es_model_name    = ".elser_model_2_linux-x86_64"
    min_confidence = 10
    num_results =5

    input_query= {
                    "query": { "text_expansion": {
                "tokens": {
                    "model_id": es_model_name,
                    "model_text": question,
                    }
                }
            }}
    response = es.search(
            index=knapi_index,
            body=input_query,
            scroll='5m',  # Set the scroll timeout (e.g., 5 minutes)
            size=3  # Set the number of documents to retrieve per scroll
            )
    all_hits = response['hits']['hits']
    print(len(all_hits))
    flag = False

    return all_hits

def reranker(query, hits):
    from sentence_transformers import CrossEncoder
    
    # To refine the results, we use a CrossEncoder
    cross_encoder_model = CrossEncoder("cross-encoder/stsb-roberta-base")
    # cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

    # Now, do the re-ranking with the cross-encoder
    sentence_pairs = [[query, hit["_source"]["chunk_text"]] for hit in hits]
    similarity_scores = cross_encoder_model.predict(sentence_pairs,activation_fct=nn.Sigmoid())
    # similarity_scores = cross_encoder_model.predict(sentence_pairs,activation_fct=nn.Sigmoid())
    
    for idx in range(len(hits)):
        hits[idx]["cross-encoder_score"] = similarity_scores[idx]

    # Sort list by CrossEncoder scores
    hits = sorted(hits, key=lambda x: x["cross-encoder_score"], reverse=True)

    for hit in hits:
        print("\t{:.3f}\t{}".format(hit["cross-encoder_score"], hit["_source"]["chunk_text"]))
        # st.info("cross-encoder_score")
        # st.write(hit["cross-encoder_score"])
        # st.write(hit["_source"]["chunk_text"])
    top_hits=hits[:3]
    print("\n\n========\n")
    return top_hits
def main():
    st.title("Search Juniper Docs")
    # Input: User enters search query
    search_query = st.text_input("Enter your search query")
    #

# sparse_index_name = "search-juniper-chunked-sparce-test_2k"
# knn_index_name = "search-juniper-chunked-dense-test_2k"
# knapi_index="juniper-knowledgebase-api-v2"
# jdc_index="search-juniper-documentation-chunked"
    # search_method = st.radio("Select Index method:", ["Dense Index(search-juniper-chunked-dense-test_2k)", "Sparse Index(search-juniper-chunked-sparce-test_2k)","juniper-documentation-chunked","knowledgebase-api-v2"])
    # search_method = st.radio("Select Index method:", ["Dense Index", "Sparse Index"])
    search_method = st.radio("Select Index method:", ["Sparse Index"])

    #Button: User triggers the search
    if st.button("Search"):
        if search_query:
            # Perform the search and get results
            st.divider()
            
            if search_method == "Dense Index":
                search_time_start = time.time()
                results = search_knn(search_query)
                search_time_end = time.time()
                search_elapsed_time = search_time_end - search_time_start
                st.info(f"Time taken to Search: {search_elapsed_time:.2f} seconds")
                context =""
                for result in results:
                    print(result["_score"])
                    print(result["_source"]["title"])
                    # print(result["_source"]["chunk_no"])
                    print(result["_source"]["main_content"])
                    if '_source' in result:
                        try:
                            context = context + "" + "**" + str(result['_source']['title']) +"** "+ "\n\n"
                            context  = context + "" + str(result['_source']['main_content']) + "\n\n"
                            context  = context + "" + str(result['_source']['url']) + "\n\n"
                        except Exception as e:
                            print(e)
            elif search_method == "Sparse Index":
                search_time_start = time.time()
                results = search_es(search_query)
                search_time_end = time.time()
                search_elapsed_time = search_time_end - search_time_start
                st.info(f"Time taken to Search: {search_elapsed_time:.2f} seconds")
                top_hits=reranker(search_query,results)
                context =""
                for result in top_hits:
                    print(result["_score"])
                    print(result["_source"]["title"])
                    # print(result["_source"]["chunk_no"])
                    print(result["_source"]["chunk_text"])
                    if '_source' in result:
                        try:
                            
                            context = context + "**" + str(result['_source']['title']) +"** "+ "\n\n"
                            context = context + "Relevance:"+ "**" + str(result['cross-encoder_score'])+ "**"+"\n\n"
                            context = context + str(result['_source']['chunk_text']) + "\n\n"
                            context  = context + "" + str(result['_source']['url']) + "\n\n"
                        except Exception as e:
                            print(e)
            # elif search_method == "juniper-documentation-chunked":
            #     search_time_start = time.time()
            #     results = search_es_jdc(search_query)
            #     search_time_end = time.time()
            #     search_elapsed_time = search_time_end - search_time_start
            #     st.info(f"Time taken to Search: {search_elapsed_time:.2f} seconds")
            #     context =""
            #     st.write(results)
            #     for result in results:
            #         print(result["_score"])
            #         print(result["_source"]["title"])
            #         # print(result["_source"]["chunk_no"])
            #         print(result["_source"]["passages"][0]['text'])
            #         if '_source' in result:
            #             try:
            #                 context = context + "**" + str(result['_source']['title']) +"** "+ "\n\n"
            #                 context = context + str(result['_source']["passages"][0]['text']) + "\n\n"
            #             except Exception as e:
            #                 print(e)

            # elif search_method == "knowledgebase-api-v2":
            #     search_time_start = time.time()
            #     results = search_es_knapi(search_query)
            #     search_time_end = time.time()
            #     search_elapsed_time = search_time_end - search_time_start
            #     st.info(f"Time taken to Search: {search_elapsed_time:.2f} seconds")
            #     context =""
            #     st.write(results)
            #     for result in results:
            #         print(result["_score"])
            #         # print(result["_source"]["title"])
            #         # print(result["_source"]["chunk_no"])
            #         print(result["_source"]["Text"])
            #         if '_source' in result:
            #             try:
            #                 # context = context + "**" + str(result['_source']['title']) +"** "+ "\n\n"
            #                 context = context + str(result['_source']['Text']) + "\n\n"
            #             except Exception as e:
            #                 print(e)
            with st.expander("References", expanded=False):
                if len(context):
                    st.write(context)

            llm_input=build_prompt(search_query,context)
            # st.write(llm_input)
            llm_start_time = time.time()
            llm_response=send_to_watsonxai(llm_input)
            llm_end_time = time.time()
            llm_elapsed_time = llm_end_time - llm_start_time
            st.info(f"Time taken to LLM response: {llm_elapsed_time:.2f} seconds")
            st.write(llm_response)

            # st.divider()
            
            # Simialrity Calculating
            # # Text input for response
            # #llm_response="The user is receiving a splash message to contact support for provisioning the Premium Analytics dashboard because the provisioning process has not been completed yet. Although the user has a Premium Analytics subscription, the provisioning process takes 24-48 hours to complete. This is a standard process for new subscriptions, and the user needs to wait for the provisioning to finish before they can access the Premium Analytics dashboard."
            # golden_response = st.text_area("Enter Ground Truth response:", "")
            # # Submit button
            # if st.button("Submit"):
            #     # Compute embeddings for input and response passages
            #     input_embedding = compute_sentence_embeddings(st.session_state.llm_response)
            #     response_embedding = compute_sentence_embeddings(st.session_state.golden_response)
            #     # Compute similarity between embeddings
            #     similarity_score_2 = compute_similarity(input_embedding, response_embedding)
            #     st.write(f"Torch  Similarity Score: {similarity_score_2:.2f}")
                    
if __name__ == "__main__":
    main()

