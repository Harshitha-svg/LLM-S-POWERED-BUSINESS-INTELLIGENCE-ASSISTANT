import pandas as pd
from groq_client import GroqClient
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
groq = GroqClient()

def create_embeddings(df: pd.DataFrame, namespace: str = "dataset"):
    """Convert text data to embeddings and store in Groq/vector DB."""
    texts = []
    for col in df.columns:
        for val in df[col].astype(str).tolist()[:200]:
            texts.append(f"{col}: {val}")
    embeddings = model.encode(texts).tolist()
    items = [{"id": str(i), "embedding": embeddings[i], "metadata": {"text": texts[i]}} for i in range(len(texts))]
    groq.upsert_vectors(namespace, items)
    return len(items)

def query_embeddings(query: str, top_k: int = 5):
    q_emb = model.encode([query])[0].tolist()
    results = groq.query(q_emb, top_k=top_k)
    return results
