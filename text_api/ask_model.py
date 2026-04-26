import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json

INDEX_PATH = "models/faiss.index"
META_PATH = "models/faiss_meta.json"

# ================= LOAD =================
print("Loading FAISS model...")

index = faiss.read_index(INDEX_PATH)
df = pd.read_json(META_PATH)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ================= SEARCH =================
def ask_model(query: str):
    query_vec = model.encode([query]).astype("float32")

    D, I = index.search(query_vec, 1)

    idx = I[0][0]

    result = df.iloc[idx].to_dict()

    return {
        "query": query,
        "result": result
    }

# ================= TEST =================
if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        if q == "exit":
            break
        print(ask_model(q))