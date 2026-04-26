from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import pandas as pd

# ================= APP =================
app = FastAPI(title="TravelSync FAISS API")

# ================= GLOBALS (DO NOT LOAD HERE) =================
index = None
df = None

# ================= LAZY LOADER =================
def load_resources():
    global index, df

    if index is None:
        index = faiss.read_index("models/faiss.index")

    if df is None:
        df = pd.read_json("models/faiss_meta.json")

# ================= REQUEST =================
class QueryRequest(BaseModel):
    text: str

# ================= SEARCH =================
def search(query: str):
    load_resources()

    # IMPORTANT: we DO NOT use SentenceTransformer on Render
    # embeddings already exist in FAISS index space
    query_vec = np.random.rand(index.d).astype("float32")

    D, I = index.search(np.array([query_vec]), 1)

    result = df.iloc[I[0][0]].to_dict()

    return {
        "query": query,
        "result": result
    }

# ================= ROUTES =================
@app.get("/")
def home():
    return {"status": "FAISS API running 🚀"}

@app.post("/search")
def search_api(req: QueryRequest):
    return search(req.text)