import pandas as pd
import json
import os

DATA_PATH = r"C:\Graduation Project\TravelSync\data\Egypt_Heritage_Artifacts_1000.csv"
OUTPUT_PATH = "TravelSync/data/passages.jsonl"

df = pd.read_csv(DATA_PATH)

# adjust column names safely
df.columns = [c.strip() for c in df.columns]

# choose correct text column
text_column = "Name" if "Name" in df.columns else df.columns[0]

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for i, row in df.iterrows():
        obj = {
            "text": str(row[text_column]),
            "label": i
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ passages.jsonl rebuilt successfully")