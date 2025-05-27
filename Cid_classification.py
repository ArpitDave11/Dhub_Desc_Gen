# Databricks notebook source
!pip install -qU langchain-community langchain-openai faiss-gpu sqlalchemy

# COMMAND ----------
import os
import json
import time
import numpy as np
import pandas as pd
import urllib.parse
import psycopg2
from sqlalchemy import create_engine, text

# LangChain & Azure OpenAI
import langchain_community
import langchain_openai
import faiss
from uuid import uuid4
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA

# ========== 1) Setup Azure OpenAI Environment ==========
os.environ["OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"
os.environ["ENDPOINT_URL"] = "https://YOUR-RESOURCE-NAME.openai.azure.com/"
os.environ["DEPLOYMENT_NAME"] = "o3-mini"
os.environ["OPENAI_API_VERSION"] = "2025-01-01-preview"

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
if not endpoint or not deployment or not openai_api_key or not api_version:
    raise EnvironmentError("Missing Azure OpenAI environment variables.")

# Map env vars to those expected by AzureChatOpenAI
os.environ["AZURE_OPENAI_API_KEY"] = openai_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
os.environ["AZURE_OPENAI_API_VERSION"] = api_version

# ========== 2) Initialize Embeddings & LLM ==========
embedding_model = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002")
llm = AzureChatOpenAI(
    azure_deployment="gpt-4-1",  # or your own deployment name
    api_version="2025-01-01-preview", 
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ========== 3) Prepare Vector Store (FAISS) For Retrieval ==========
# If you already have your FAISS index built and saved locally, load it here.
# Otherwise, build from your JSON of documents & embeddings. 
# (Below is a minimal example showing how to load an existing faiss_index.)

faiss_index_path = "/dbfs/mnt/genai/knowledge_base/faiss_index"
new_vector_store = None
try:
    new_vector_store = faiss.read_index(faiss_index_path + "/index.faiss")
    # If you used LangChain's FAISS wrapper:
    vector_store = langchain_community.vectorstores.FAISS.load_local(
        faiss_index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index from {faiss_index_path}: {e}")

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ========== 4) Pull Data from Postgres & Generate AI Descriptions ==========
host = "z34764-pfeyus2dhubfts1-dev.postgres.database.azure.com"
database = "wma_dataproduct_metadata"
user = "postgres"
password = "Defabr1c"
port = 5432

encoded_user = urllib.parse.quote_plus(user)
encoded_password = urllib.parse.quote_plus(password)
connection_str = f"postgresql+psycopg2://{encoded_user}:{encoded_password}@{host}:{port}/{database}"

engine = create_engine(connection_str)
conn = engine.connect()

sql_query = text("""
    SELECT schema_fact_key, CAST(schema_id AS VARCHAR) AS schema_id, 
           schema_name, schema_type, schema_description, 
           json_schema_location, schema_at_ubs_url,
           created_at, start_date, end_date, 
           current_version_flag, last_updated_by
    FROM wma_dataproduct_metadata.schemas
""")
df = pd.read_sql(sql_query, conn)
conn.close()

# We'll define a function to query the RAG chain:
def fetch_description(name: str):
    """
    Use the RetrievalQA chain to fetch a factual description of 'name'.
    """
    query = (
        f"Given the attribute name: {name}\n"
        "Provide a concise, factual description of this attribute, "
        "tailored with its meaning, and a relevant example of its use in banking. "
        "Output only the description."
    )
    try:
        result = qa_chain.run(query)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def get_result_only(name: str):
    """
    Simple wrapper to handle the QA chain dict or string output.
    """
    response = fetch_description(name)
    # If the chain returns a string directly, just return it
    return response

df["Ai_generated_text"] = df["schema_name"].apply(get_result_only)

# ========== 5) Classification Step (CID/Non-CID & RAG) ==========

CLASSIFICATION_RULES = r"""
You are given a piece of text (an AI-generated description). 
You must classify it according to the following rules:

----
1) Determine if it is "CID" or "Non-CID":

   - CID (Client Identifying Data) includes:
       * Full name, address, DOB, account number, govt-issued ID, contact details
       * Or any combination that could identify an individual
   - Non-CID includes:
       * Aggregated/anonymized data
       * Generic info not tied to an individual

2) Determine its RAG classification (Red, Amber, or Green):

   - Red = High Sensitivity (all CID is typically Red)
   - Amber = Medium Sensitivity (often internal IDs that do not directly identify client)
   - Green = Low Sensitivity (non-CID or aggregated data)

3) Provide a short justification for both decisions.

4) Detailed categories for reference:
   - Category A (Direct CID): name, address, passport ID => Red
   - Category B (Indirect but still CID, e.g., account number) => Red
   - Category C (combinations that can identify) => Red if combined
   - Category D (non-sensitive internal IDs) => Non-CID => Amber
   - Otherwise => Non-CID => Green
----
Your Output Format (valid JSON):
{
  "CID_Status": "CID" or "Non-CID",
  "RAG_Category": "Red" or "Amber" or "Green",
  "Justification": "Concise reason"
}
"""

def classify_text_with_llm(text: str) -> dict:
    """
    Sends the text to AzureChatOpenAI with the classification rules,
    asks for a JSON response specifying:
      1) "CID_Status"
      2) "RAG_Category"
      3) "Justification"
    """
    system_message = (
        "You are a classification assistant. "
        "Follow the classification instructions carefully and output valid JSON."
    )

    user_message = f"""
    Classification Rules:
    {CLASSIFICATION_RULES}

    Text to classify:
    {text}
    """

    classification_llm = AzureChatOpenAI(
        azure_deployment="gpt-4-1",  # same or separate deployment
        api_version=api_version,
        temperature=0
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    response = classification_llm(messages)
    content = response.content.strip()

    try:
        import json
        result = json.loads(content)
        required_keys = {"CID_Status", "RAG_Category", "Justification"}
        if not required_keys.issubset(result.keys()):
            raise ValueError("Missing required keys in JSON response.")
        return result
    except Exception as e:
        return {
            "CID_Status": "Unknown",
            "RAG_Category": "Unknown",
            "Justification": f"Parsing error or invalid format. Raw: {content}"
        }

# Apply classification to each row's Ai_generated_text
classification_results = df["Ai_generated_text"].apply(classify_text_with_llm)

df["CID_Status"] = classification_results.apply(lambda x: x["CID_Status"])
df["RAG_Category"] = classification_results.apply(lambda x: x["RAG_Category"])
df["Justification"] = classification_results.apply(lambda x: x["Justification"])

# ========== 6) Display or Save the Final Result ==========

df_display_cols = ["schema_name", "Ai_generated_text", "CID_Status", "RAG_Category", "Justification"]
display(df[df_display_cols])
