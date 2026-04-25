import pandas as pd
import json

INPUT_FILE = r"C:\Graduation Project\TravelSync\data\Egypt_Heritage_Artifacts_1000.csv"
OUTPUT_FILE = "data/qa_dataset.json"

print("Loading CSV...")
df = pd.read_csv(INPUT_FILE)

data = []

for _, row in df.iterrows():

    title = str(row["Name"])
    context = str(row["Description"])

    if len(context) < 50:
        continue

    question = f"What is {title}?"

    answer_text = context.split(".")[0]  # First sentence only
    answer_start = context.find(answer_text)

    example = {
        "context": context,
        "question": question,
        "answers": {
            "text": [answer_text],
            "answer_start": [answer_start]
        }
    }

    data.append(example)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ QA dataset created successfully.")