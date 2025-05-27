# src/rag.py

import os
import csv
import json
import numpy as np
import faiss
import openai
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4-1")
EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002")

openai.api_type = "azure"
openai.api_key = OPENAI_API_KEY
openai.api_base = AZURE_ENDPOINT
openai.api_version = OPENAI_API_VERSION

def build_or_load_faiss(csv_path: str, faiss_dir: str):
    """
    Build a FAISS index from the CSV of embeddings if not exists,
    otherwise load from disk.
    """
    if not os.path.exists(faiss_dir):
        os.makedirs(faiss_dir, exist_ok=True)

    faiss_index_file = os.path.join(faiss_dir, "index.faiss")
    faiss_store_file = os.path.join(faiss_dir, "faiss_store.pkl")

    # If the index and mapping already exist, we load them
    if os.path.exists(faiss_index_file) and os.path.exists(faiss_store_file):
        print("Loading existing FAISS index...")
        embeddings = AzureOpenAIEmbeddings(deployment=EMBED_DEPLOYMENT)
        vector_store = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        return vector_store

    # Otherwise, build from CSV
    print("Building FAISS index from CSV embeddings...")
    documents = []
    embeddings_list = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model_name"]
            entity_name = row["entity_name"]
            attribute_name = row["attribute_name"]
            embedding_str = row["embedding"]

            # Convert the embedding from JSON string to list[float]
            embedding = json.loads(embedding_str)
            embeddings_list.append(embedding)

            metadata = {
                "model_name": model_name,
                "entity_name": entity_name,
            }
            doc = Document(page_content=attribute_name, metadata=metadata)
            documents.append(doc)

    embeddings_array = np.array(embeddings_list, dtype="float32")
    dim = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)

    ids = [str(uuid4()) for _ in documents]
    docstore = InMemoryDocstore({uid: doc for uid, doc in zip(ids, documents)})
    index_to_id = {i: uid for i, uid in enumerate(ids)}

    # Create FAISS VectorStore
    azure_embeddings = AzureOpenAIEmbeddings(deployment=EMBED_DEPLOYMENT)
    vector_store = FAISS(
        embedding_function=azure_embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_id
    )

    # Save it
    vector_store.save_local(faiss_dir)
    return vector_store

def get_qa_chain(vector_store: FAISS):
    """Returns a RetrievalQA chain using GPT-based chat model."""
    llm = AzureChatOpenAI(
        azure_deployment=DEPLOYMENT_NAME,
        api_version=OPENAI_API_VERSION,
        temperature=0
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return chain

def main():
    # Example usage
    csv_path = os.path.join("data", "attributes_enriched.csv")  # or wherever you stored it
    faiss_dir = "faiss_index"

    vector_store = build_or_load_faiss(csv_path, faiss_dir)
    qa_chain = get_qa_chain(vector_store)

    # Example query
    print("\n=== Testing a sample query ===")
    user_query = "What does 'Loan Amount' attribute represent?"
    response = qa_chain.run(user_query)
    print("Q:", user_query)
    print("A:", response)

if __name__ == "__main__":
    main()
