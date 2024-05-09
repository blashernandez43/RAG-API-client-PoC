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

Wait for the build to complete and access the public URL by selecting the **Domain mappings** tab of the open **Application** pane.  Or go into the project by selecting **Projects** from the **Code Engine** side menu. Open the project, then select **Applications**. You will see a URL link under the **Application Link**.
