# src/retriever.py

from typing import List, Dict, Tuple
import pandas as pd

class HybridRetriever:
    """
    Hybrid retriever that:
    1) Uses vectorstore to retrieve top-k contextual rows (semantic search)
    2) Uses pandas to compute precise stats for requested filters
    """

    def __init__(self, vectorstore, df: pd.DataFrame):
        self.vs = vectorstore
        self.df = df

    def retrieve(self, query: str, k: int = 5, filters: Dict = None) -> Tuple[List, Dict]:
        """
        Retrieve documents and compute filtered statistics.

        Args:
            query (str): Natural language query
            k (int): Number of vectorstore matches
            filters (Dict): Optional filters (e.g., {"Region": "South", "Product": "Widget C"})

        Returns:
            docs (list): Retrieved documents from vectorstore
            stats (dict): Filtered statistics from DataFrame
        """
        # semantic search from vectorstore
        docs = self.vs.similarity_search(query, k=k)

        # filter df if filters provided
        df_filtered = self.df.copy()
        if filters:
            for key, value in filters.items():
                if key in df_filtered.columns and value is not None:
                    df_filtered = df_filtered[df_filtered[key] == value]

        # compute precise stats on filtered df
        stats = {
            "total_sales": float(df_filtered['Sales'].sum()),
            "avg_sales": float(df_filtered['Sales'].mean()) if len(df_filtered) > 0 else 0,
            "transactions": int(len(df_filtered)),
            "avg_customer_age": float(df_filtered['Customer_Age'].mean()) if len(df_filtered) > 0 else 0,
            "avg_satisfaction": float(df_filtered['Customer_Satisfaction'].mean()) if len(df_filtered) > 0 else 0,
        }

        return docs, stats
