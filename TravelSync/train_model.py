import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = r"C:\Graduation Project\TravelSync\data\passages.jsonl"
INDEX_PATH = r"C:\Graduation Project\TravelSync\models\knowledge.index"
META_PATH = r"C:\Graduation Project\TravelSync\models\knowledge_meta.jsonl"

print("Loading multilingual embedding model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

passages = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        passages.append(json.loads(line))

print("Embedding text...")
texts = [p["text"] for p in passages]
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

print("Building vector index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "w", encoding="utf-8") as f:
    for p in passages:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print("✅ AI model built successfully.")