import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import gdown

MODEL_PATH = "models/classifier"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading text model...")

    os.makedirs("models", exist_ok=True)

    gdown.download_folder(
        "https://drive.google.com/drive/folders/1VLboJGUbPsoJf7zNZuxn0d3CK_8dSoQc?usp=drive_link",
        output=MODEL_PATH,
        quiet=False
    )

# ----------------------------
# LOAD MODEL (runs once)
# ----------------------------
print("Loading text model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

with open("models/classifier/labels.json", "r", encoding="utf-8") as f:
    id_to_label = json.load(f)

df = pd.read_csv(r"C:\Graduation Project\TravelSync\data\Egypt_Heritage_Artifacts_1000.csv")


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def ask_model(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits).item()
    predicted_name = id_to_label[str(predicted_class)]

    result = df[df["Name"] == predicted_name]

    if result.empty:
        return {
            "error": "No information found",
            "prediction": predicted_name
        }

    row = result.iloc[0]

    return {
        "prediction": predicted_name,
        "data": row.to_dict()
    }


# ----------------------------
# OPTIONAL: LOCAL TESTING
# ----------------------------
if __name__ == "__main__":
    print("\n✅ AI Ready. Type 'exit' to quit.\n")

    while True:
        question = input("Ask: ")

        if question.lower() == "exit":
            break

        result = ask_model(question)
        print(result)