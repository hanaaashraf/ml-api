import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= PATHS =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "data", "passages.jsonl")
INDEX_PATH = os.path.join(BASE_DIR, "models", "knowledge.index")
META_PATH = os.path.join(BASE_DIR, "models", "knowledge_meta.jsonl")

# ================= SAFETY CHECK =================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Missing dataset file: {INPUT_FILE}\n"
        "Create passages.jsonl or rebuild it from CSV."
    )

# ================= MODEL =================

print("Loading multilingual embedding model...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ================= LOAD DATA =================

passages = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        passages.append(json.loads(line))

# ================= EMBEDDINGS =================

print("Embedding text...")
texts = [p["text"] for p in passages]

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# ================= BUILD INDEX =================

print("Building vector index...")

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

# ================= SAVE META =================

with open(META_PATH, "w", encoding="utf-8") as f:
    for p in passages:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print("✅ AI model built successfully.")