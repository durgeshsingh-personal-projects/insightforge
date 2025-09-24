# src/evaluation.py

import os
from src.data_prep import load_data
from src.knowledge_base import load_vectorstore
from src.retriever import HybridRetriever
from src.rag_chain import build_rag_chain, run_rag_answer
from langchain.evaluation.qa import QAEvalChain  # updated import
from langchain_community.llms import OpenAI  # updated import for LLM

# Load sales data
DATA_PATH = "data/sales_data.csv"
df = load_data(DATA_PATH)

# Load or build vectorstore
vs = load_vectorstore()

# Initialize retriever and RAG chain
retriever = HybridRetriever(vs, df)
chain = build_rag_chain(vs)

# Initialize evaluation chain
llm_eval = OpenAI(temperature=0)
qa_eval = QAEvalChain.from_llm(llm_eval)

# Define evaluation examples
examples = [
    {
        "query": "How many units of Widget C sold on 2022-01-01?",
        "answer": "786",
    },
    {
        "query": "What region sold the most Widget D on 2022-01-02?",
        "answer": "East",
    },
]

# Load vectorstore (already persisted)
vs = load_vectorstore()

# Build RAG chain
chain = build_rag_chain(vs)

# Run RAG answers for all examples
predictions = []
for ex in examples:
    result = run_rag_answer(chain, ex["query"])  # likely returns tuple
    if isinstance(result, tuple):
        answer = result[0]  # take the first element as the text answer
    else:
        answer = result

    print(f"Q: {ex['query']}")
    print(f"Predicted: {answer}")
    print(f"Expected: {ex['answer']}\n")
    predictions.append(answer)

# Evaluate manually
print("Manual QA evaluation results:")
for ex, pred in zip(examples, predictions):
    correct = ex["answer"].lower() in pred.lower()
    print(f"Query: {ex['query']}")
    print(f"Prediction: {pred}")
    print(f"Expected: {ex['answer']}")
    print(f"Correct? {'✅' if correct else '❌'}\n")
