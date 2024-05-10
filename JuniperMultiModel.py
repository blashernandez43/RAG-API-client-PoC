import time
from collections import defaultdict, Counter
import re
import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize

class JuniperMultiModel:

    def __init__(self):
        model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self._sentence_model = model

    def retrieve_documents(self,question, es_elser_client, es_dense_client):
        ## elastic search retriever one, ELSER model
        tic = time.perf_counter()
        index_name = 'search-juniper-sparse-99k-50overlap'
        es_output_elser = self.es_query_process_parse_elser(
            question, num_results=10, 
            index_name=index_name, 
            es_model_name=".elser_model_2_linux-x86_64",
            es_client=es_elser_client
        )

        final_es_output_elser = self.process_es_output_elser(es_output_elser)
        cleaned_es_output_elser = self.clean_text(final_es_output_elser)

        ## elastic search retriever two, Dense vector
        index_name = 'search-juniper-dense-minilm_l6_512-99k'
        es_output_dense = self.esqueryprocess_dense(
            question, 
            num_results=10, 
            index_name=index_name, 
            model=self._sentence_model,
            es_client=es_dense_client)

        final_es_output_dense = self.process_es_output_dense(es_output_dense)
        toc = time.perf_counter()
        return cleaned_es_output_elser, final_es_output_dense

    def clean_text(self, text):
        # Remove newline characters along with other special characters and multiple spaces
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with a single space
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()

    def es_query_process_parse_elser(self, 
                                     question, 
                                     num_results, 
                                     index_name, 
                                     es_client,
                                     es_model_name=".elser_model_2_linux-x86_64"
                                     ):
        min_confidence = 10
        
        # Define the query for Elasticsearch
        query_body = {
            "query": {
                "bool": { 
                    "should": [
                        {
                            "text_expansion": {
                                "chunk_text": {
                                    "model_text": question,
                                    "model_id": es_model_name
                                }
                            }
                        }
                    ],
                    "must": {
                        "multi_match": {
                            "query": question,
                            "type": "best_fields",
                            "fields": ["title", "body_content"],  # Including meta-description
                            "tie_breaker": 0.9
                        }
                    }
                }
            },
            "size": num_results,
            "min_score": min_confidence
        }

        # Execute the query
        try:
            response = es_client.search(index=index_name, body=query_body)
            document_id = [(hit["_id"], hit["_id"]) for hit in response["hits"]["hits"]]
            # Extract relevant information from hits
            relevant_chunks = [(hit["_source"], hit["_score"]) for hit in response["hits"]["hits"]]
            return relevant_chunks
        except Exception as e:
            return {"msg": "Error searching indexes", "error": str(e)}

    def esqueryprocess_dense(self, 
                             question,
                             num_results,
                             index_name, 
                             model,
                             es_client
                             ):
        try:
            question_embedding = model.encode(question)
            search_query={
            "query": {
            "multi_match" : { ##REMOVE MULTI_MATCH for non-hybrid queries
                "query":question,
                    "type":"best_fields",
                    "fields":[ "body_content", "title"], 
                    # "fields": ["main_content", "title"],
                    # "fields":[ "body_content", "title", "meta-description"], 
                    "tie_breaker": 0.3

                }
            },
            "knn": {
                "field": "content_embedding",
                "query_vector": question_embedding,
                "k": 10,
                "num_candidates": 100,
                "boost": 10
            },
            }
            query_nested_index = es_client.search(
            index=index_name,
            body=search_query,
            scroll='5m',  # Set the scroll timeout (e.g., 5 minutes)
            size=num_results  # Set the number of documents to retrieve per scroll
            )
        except Exception as e:
            return {"msg": "Error searching indexes", "error": e}
    
        # Get relevant chunks and format
        references_context1 = [(chunks["_source"], chunks["_score"]) for chunks in query_nested_index["hits"]["hits"]]
        return references_context1

    def tokenize(self,text):
        # Simple tokenization based on whitespace; replace with more sophisticated tokenizer if needed
        return word_tokenize(text)

    def process_es_output_elser(self, es_output, content_field='main_content'):
        content_by_title = defaultdict(str)
        title_frequency = Counter()

        # Collect `main_content` or `chunk_text` for each title and count frequencies
        for output in es_output:
            title = output[0]['title']
            content = output[0][content_field]  # Dynamically select the content field
            content_by_title[title] += content + " "
            title_frequency[title] += 1

        # Determine unique titles
        unique_titles = set(title_frequency.keys())

        # Concatenate content based on the unique title count
        final_content = ""
        if len(unique_titles) == 1:
            # If there's only one unique title, use the first document's content
            first_title = next(iter(unique_titles))
            final_content = content_by_title[first_title]
        else:
            # Concatenate all content if multiple unique titles exist
            for title in unique_titles:
                final_content += content_by_title[title]

        # Count tokens and truncate if necessary
        tokens = self.tokenize(final_content)
        if len(tokens) > 23000:
            final_content = ' '.join(tokens[:23000])  # Truncate to 30000 tokens

        # Print the number of tokens in the final content
        print("Number of tokens in final content:", len(self.tokenize(final_content)))

        return final_content
    
    def process_es_output_dense(self,es_output, content_field='main_content'):
        content_by_title = defaultdict(str)
        title_frequency = Counter()

        # Collect `main_content` or `chunk_text` for each title and count frequencies, truncating each to 3000 tokens
        for output in es_output:
            title = output[0]['title']
            content = output[0][content_field]  # Dynamically select the content field
            # Clean and truncate each content to 3000 tokens
            cleaned_content = self.clean_text(content)
            tokens = self.tokenize(cleaned_content)
            if len(tokens) > 3000:
                cleaned_content = ' '.join(tokens[:3000])
            content_by_title[title] += cleaned_content + " "
            title_frequency[title] += 1

        # Determine unique titles
        unique_titles = set(title_frequency.keys())

        # Concatenate content based on the unique title count, respecting the total token limit of 25000
        final_content = ""
        for title in unique_titles:
            additional_tokens = self.tokenize(content_by_title[title])
            final_tokens = self.tokenize(final_content)
            # Add only up to the limit
            if len(final_tokens) + len(additional_tokens) > 23000:
                remaining_tokens = 25000 - len(final_tokens)
                final_content += ' '.join(additional_tokens[:remaining_tokens])
                break
            else:
                final_content += ' '.join(additional_tokens) + " "

        # Final token count and truncation check (should not exceed, but safe to double-check)
        final_tokens = self.tokenize(final_content)
        if len(final_tokens) > 25000:
            final_content = ' '.join(final_tokens[:23000])

        # Print the number of tokens in the final content
        print("Number of tokens in final content:", len(self.tokenize(final_content)))

        return final_content

    def build_prompt(self, user_query,context,model_id="MIXTRAL"):

        SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        USER_PROMPT = """A user has asked the following question: '{question}'. Use the information available from the '{context}' to provide a detailed and accurate answer in less than 150 words. If necessary, cite relevant sources or data to support your response. Ensure your answer is clearly structured and directly addresses the question.
        If the answer to the question is not available in the any of the provided context, please say: "I don't know the answer to that question." 
        While you have to answer only from the context, the answer language must be in the way such that it is coming from your own knowledge.
        """

        LLAMA3_PROMPT= """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Answer Based on the Provided Context: 
        """

        MIXTRAL_PROMPT = """[INST]
        [ROLE]
        {system_prompt}
        [/ROLE]
        [USER_INSTRUCTIONS]
        {user_prompt}
        [/USER_INSTRUCTIONS]

        Answer Based on the Provided Context:
        [/INST]"""

        user_prompt = USER_PROMPT.format(question=user_query,context=context)
        if  model_id == "MIXTRAL":
            formatted_prompt = MIXTRAL_PROMPT.format(system_prompt=SYSTEM_PROMPT,user_prompt=user_prompt)
        elif model_id == "LLAMA3":
            formatted_prompt = LLAMA3_PROMPT.format(system_prompt=SYSTEM_PROMPT,user_prompt=user_prompt)
        return formatted_prompt


    async def send_to_watsonxai(self,
                        prompts,
                        type="",
                        model_id="MIXTRAL",
                        decoding_method="greedy",
                        max_new_tokens=500,
                        min_new_tokens=30,
                        temperature=1.0,
                        repetition_penalty=1.0,
                        wml_credentials={}
                        ):
        

        print(f"============== searchtype {type} model {model_id}")
        tic = time.perf_counter()
        if  model_id == "MIXTRAL":
            model_name = "mistralai/mixtral-8x7b-instruct-v01"
        elif model_id == "LLAMA3":
            model_name="meta-llama/llama-3-70b-instruct"

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
            credentials=wml_credentials,
            project_id='8322329b-6d42-41c7-a158-c74d429a4c3b')

        response=model.generate_text(prompts)
        toc = time.perf_counter()
        duration = toc - tic

        return {
            "response": response,
            "model_id": model_name,
            "query_type": type,
            "model_load_time": duration
        }