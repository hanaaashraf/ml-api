import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ================= PATH =================
DATA_PATH = r"C:\Graduation Project\text_api\data\Egypt_Heritage_Artifacts_1000.csv"

INDEX_PATH = "models/faiss.index"
META_PATH = "models/faiss_meta.json"

os.makedirs("models", exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

# fill missing values FIRST
df = df.fillna("")

df["text"] = (
    df["Name"].astype(str) + " " +
    df["Type"].astype(str) + " " +
    df["Detailed Historical Background"].astype(str)
)

texts = df["text"].tolist()

# ================= MODEL =================
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Encoding texts...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# ================= FAISS INDEX =================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

# ================= SAVE METADATA =================
df.to_json(META_PATH, orient="records", force_ascii=False, indent=2)

print("✅ FAISS index created successfully")