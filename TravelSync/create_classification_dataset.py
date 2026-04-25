import pandas as pd
import json

INPUT_FILE = r"C:\Graduation Project\TravelSync\data\Egypt_Heritage_Artifacts_1000.csv"
OUTPUT_FILE = r"C:\Graduation Project\TravelSync\data\classification_dataset.json"

df = pd.read_csv(INPUT_FILE)

data = []

for _, row in df.iterrows():

    name = str(row["Name"])

    # Create multiple question variations for better learning
    questions = [
        f"What is {name}?",
        f"Tell me about {name}",
        f"Information about {name}",
        f"Explain {name}"
    ]

    for q in questions:
        data.append({
            "text": q,
            "label": name
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ Classification dataset created.")