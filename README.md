# Watsonx Discovery & Watsonx.ai RAG Application

This README will guide you through the steps to install the project locally or via IBM Code Engine. Additionally, you will learn how to access the Swagger documentation once the project is deployed.

## How to Install Locally

To install this project locally, follow these steps:

1. **Clone the repository**k

    ```
    git clone https://github.com/blashernandez43/RAG-API-client-PoC.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd RAG-API-client-PoC
    git checkout poc-streamlit-app
    ```

3. **Create the enviroment, activate it, and install Requirements:**

    ```bash
    python -m venv assetEnv
    source assetEnv/bin/activate
    python -m pip install -r requirements.txt
    ```

4. **Update your secrets:**

    Copy `env` to `.env` and fill in the variables with your url, passwords, and apikeys for Elasticsearch and watsonx.ai

6. **Start the project:**

    ```bash
    streamlit run app.py
    ```

7. **application access:**

   You will see a streamlit app launch at `http://localhost:8501/`

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