import pandas as pd
import json
from langdetect import detect
from tqdm import tqdm
import re

INPUT_FILE = r"C:\Graduation Project\TravelSync\data\Egypt_Heritage_Artifacts_1000.csv"
OUTPUT_FILE = r"C:\Graduation Project\TravelSync\data\passages.jsonl"

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Try detecting columns dynamically
title_col = None
text_col = None

for col in df.columns:
    if "name" in col.lower() or "title" in col.lower():
        title_col = col
    if "description" in col.lower() or "background" in col.lower():
        text_col = col

if text_col is None:
    raise Exception("Description column not found.")

print("Processing data...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    pid = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        title = clean_text(row.get(title_col, ""))
        text = clean_text(row[text_col])

        if len(text) < 20:
            continue

        try:
            lang = detect(text)
        except:
            lang = "unknown"

        record = {
            "id": pid,
            "entity": title.lower(),
            "title": title,
            "text": text,
            "language": lang
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        pid += 1

print("✅ Preprocessing complete.")