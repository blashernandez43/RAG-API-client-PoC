# Watsonx Discovery & Watsonx.ai RAG Application

This README will guide you through the steps to install the project locally or via IBM Code Engine. Additionally, you will learn how to access the Swagger documentation once the project is deployed.

## How to Install Locally

To install this project locally, follow these steps:

1. **Clone the repository**

    ```
    git clone https://github.com/blashernandez43/RAG-API-client-PoC.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd RAG-API-client-PoC
    ```

3. **Create the enviroment, activate it, and install Requirements:**

    ```bash
    python3 -m venv assetEnv
    source assetEnv/bin/activate
    python3 -m pip install -r requirements.txt
    ```

4. **Update your secrets:**

    Copy `env` to `.env` and fill in the variables with your url, passwords, and apikeys.

6. **Start the project:**

    ```bash
    python3 app.py
    ```

7. **URL access:**

    Go to `localhost:4050` to verify that the api is running. You should see a "Hello World" message.

    To access Swagger go to http://0.0.0.0:4050/docs

## How to Deploy on Code Engine

We have created Terraform scripts to help deploy this on **IBM Cloud Code Engine** service. Make sure you have this service provisioned.

1. Clone the repo: `git clone https://github.com/ibm-build-lab/rag-codeengine-terraform-setup/tree/updatedTF`
2. Change into the cloned directory `cd rag-codeengine-terraform-setup`
3. Edit the `terraform.tfvars` file and fill in all the required values. Note for this api, the COS and WD variables are unnecessary and can be left as default.
4. Update the `variables.tf` file to change the value of `source_url` to point to `https://github.com/blashernandez43/RAG-API-client-PoC`
5. Run `terraform init` to initialize your terraform environment
6. Run `terraform plan` to see what resources will be created
7. Run `terraform apply` to create the resources

Verify that this has created a **Code Engine** project and application. 

- From the IBM Cloud search bar, search on `Code Engine` to bring up the service
- Go to `Projects` and search for the project you specified in the `terraform.tfvars` file
- Within the project you should see an application running with a `Ready` status

### Accessing the URL on Code Engine

Wait for the build to complete and access the public URL by selecting the **Domain mappings** tab of the open **Application** pane.  Or go into the project by selecting **Projects** from the **Code Engine** side menu. Open the project, then select **Applications**. You will see a URL link under the **Application Link**.
    
### How to Access Swagger Once Deployed

A quick sanity check with `<url>/docs` will take you to the swagger ui.

## Using the API

After deploying the application, you can now test the API: 

### Swagger

1. Open Swagger by going to `<url>/docs`.

2. Authenticate the `queryWXDLLM` api by clicking the lock button to the right.  Enter the value you added for the `RAG_APP_API_KEY`.

3. Click the `Try it out` button and customize your request body:
    ```
    {
      "question": "<your question>",
      "num_results": "5",        # how many results from each index should be returned
      "llm_params": {
        "model_id": "mistralai/mixtral-8x7b-instruct-v01",
        "inputs": [],
        "parameters": {
          "decoding_method": "greedy",
          "max_new_tokens": 500,
          "min_new_tokens": 1,
          "moderations": {
            "hap_input": "true",
            "hap_output": "true",
            "threshold": 0.75
          }
        }
      },
      "llm_instructions": "[INST]<<SYS>>You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Be brief in your answers. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'''t know the answer to a question, please do not share false information. <</SYS>>\nGenerate the next agent response by answering the question. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities and use the tiles of documents to separate between topics or domains. Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer.\n{context_str}<</SYS>>\n\n{query_str}. Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer. [/INST]"
    }
    ```

    At a minimum, specify:
    ```
    {
      "question": "<your question>"
    }
    ```
    All other values have defaults, you can adjust the other parameters to improve your results.
   
### cURL

To execute this api from command line, use this command: 
```
curl --location '<application url>/queryWXDLLM' \
--header 'Content-Type: application/json' \
--header 'RAG-APP-API-Key: <your custom RAG-APP-API-KEY value>' \
--data '{
  "question": "string"
}'
```
### Postman

1. Open a new tab and from the request type dropdown, select POST. In the url, paste your url (in this example, it's localhost): `http://127.0.0.1:4050/queryWXDLLM`

2. Under Authorization, choose type **API Key**, add the following key/value: `RAG-APP-API-Key`/`<value for RAG_APP_API_KEY from .env>`

3. Under Body, select `raw` and paste the following json:
```
{
  "question": "<your question>",
}
```
4. Hit the blue `SEND` button and wait for your result.

