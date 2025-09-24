# src/knowledge_base.py

import os
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.data_prep import load_data, basic_aggregates

DATA_PATH = "data/sales_data.csv"
INDEX_PATH = "vectorstore/faiss_index"


def build_documents(df: pd.DataFrame) -> list:
    """
    Convert sales DataFrame into a list of LangChain Documents.
    Each row is turned into a natural language fact string for embedding.
    """
    docs = []
    for _, row in df.iterrows():
        text = (
            f"On {row['Date'].date()}, Product {row['Product']} was sold in "
            f"the {row['Region']} region for {row['Sales']} units. "
            f"Customer was {row['Customer_Age']} years old, "
            f"{row['Customer_Gender']}, with satisfaction score "
            f"{row['Customer_Satisfaction']:.2f}."
        )
        metadata = {
            "date": str(row["Date"].date()),
            "product": row["Product"],
            "region": row["Region"],
            "sales": row["Sales"],
            "customer_age": row["Customer_Age"],
            "customer_gender": row["Customer_Gender"],
            "customer_satisfaction": row["Customer_Satisfaction"],
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def build_vectorstore(df: pd.DataFrame, persist: bool = True):
    """
    Build a FAISS vectorstore from sales data.
    If persist=True, saves the index locally for reuse.
    """
    embeddings = OpenAIEmbeddings()
    docs = build_documents(df)
    vectorstore = FAISS.from_documents(docs, embeddings)

    if persist:
        os.makedirs(INDEX_PATH, exist_ok=True)
        vectorstore.save_local(INDEX_PATH)

    return vectorstore


def load_vectorstore():
    """
    Load FAISS vectorstore if it exists, otherwise build from scratch.
    """
    embeddings = OpenAIEmbeddings()
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        df = load_data(DATA_PATH)
        return build_vectorstore(df, persist=True)


def query_knowledge_base(query: str, k: int = 5):
    """
    Query the knowledge base for top-k most relevant sales facts.
    """
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search(query, k=k)


def get_vectorstore_and_data():
    """
    Utility function to return both the vectorstore and the raw DataFrame.
    Useful for HybridRetriever.
    """
    df = load_data(DATA_PATH)
    vs = load_vectorstore()
    return vs, df


if __name__ == "__main__":
    # Quick test
    df = load_data(DATA_PATH)
    print("Building aggregates...")
    aggs = basic_aggregates(df)
    print("Top products:", aggs["sales_by_product"].head(), "\n")

    print("Building vectorstore and running test query...")
    vs = build_vectorstore(df, persist=True)
    results = query_knowledge_base("Show me sales in South region for Widget C in January 2022", k=3)

    for i, res in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("Content:", res.page_content)
        print("Metadata:", res.metadata)
