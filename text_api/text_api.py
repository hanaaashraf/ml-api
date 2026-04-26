from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ================= APP =================
app = FastAPI(title="TravelSync FAISS API")

# ================= LOAD MODEL ON START =================
print("Loading FAISS index...")

INDEX_PATH = "models/faiss.index"
META_PATH = "models/faiss_meta.json"

index = faiss.read_index(INDEX_PATH)
df = pd.read_json(META_PATH)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ================= REQUEST FORMAT =================
class QueryRequest(BaseModel):
    text: str

# ================= SEARCH FUNCTION =================
def search_faiss(query: str):
    query_vec = embedding_model.encode([query]).astype("float32")

    D, I = index.search(query_vec, 1)

    idx = I[0][0]

    result = df.iloc[idx].to_dict()

    return {
        "query": query,
        "result": result
    }

# ================= ROUTES =================

@app.get("/")
def home():
    return {"status": "FAISS API running 🚀"}

@app.post("/search")
def search(req: QueryRequest):
    return search_faiss(req.text)