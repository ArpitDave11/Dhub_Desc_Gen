# src/review_evaluation.py

import os
import csv
import json
import numpy as np
import openai
from dotenv import load_dotenv
from typing import List, Tuple
from rag import build_or_load_faiss, get_qa_chain
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    a = np.array(vec_a, dtype="float32")
    b = np.array(vec_b, dtype="float32")
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def embed_text(text: str, embed_deployment: str) -> List[float]:
    """Get an embedding for the given text using Azure OpenAI."""
    response = openai.Embedding.create(
        input=text,
        engine=embed_deployment
    )
    return response["data"][0]["embedding"]

def main():
    # 1) Build or load the RAG pipeline
    csv_path = os.path.join("data", "attributes_enriched.csv")
    faiss_dir = "faiss_index"
    vector_store = build_or_load_faiss(csv_path, faiss_dir)
    qa_chain = get_qa_chain(vector_store)

    # 2) Load the test set with ground-truth references
    test_file = "evaluation_questions.csv"
    # Format: question,reference_answer
    # e.g.
    # question,reference_answer
    # "What is loan amount?","Loan amount is the total principal sum."

    if not os.path.exists(test_file):
        print(f"ERROR: {test_file} not found.")
        return

    # 3) Evaluate each question
    embed_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002")
    results = []
    with open(test_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question"]
            reference_answer = row["reference_answer"]
            # a) get model’s answer
            llm_answer = qa_chain.run(question)

            # b) embed both answers for a simple similarity measure
            ref_embedding = embed_text(reference_answer, embed_deployment)
            llm_embedding = embed_text(llm_answer, embed_deployment)
            score = cosine_similarity(ref_embedding, llm_embedding)

            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "llm_answer": llm_answer,
                "similarity_score": score
            })
            print(f"Q: {question}\nRef: {reference_answer}\nLLM: {llm_answer}\nSimilarity: {score}\n---")

    # 4) Write out results
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Evaluation complete. Results in evaluation_results.json")

if __name__ == "__main__":
    main()
