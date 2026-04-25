import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ----------------------------
# CONFIG
# ----------------------------

MODEL_PATH = "models/classifier"

tokenizer = None
model = None
id_to_label = None
df = None

# ----------------------------
# LAZY LOADING (IMPORTANT FOR RENDER)
# ----------------------------

def load_resources():
    global tokenizer, model, id_to_label, df

    if tokenizer is None or model is None:

        print("Loading text model...")

        # Load tokenizer + model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()

        # Load labels
        with open(f"{MODEL_PATH}/labels.json", "r", encoding="utf-8") as f:
            id_to_label = json.load(f)

        # Load dataset
        df = pd.read_csv("TravelSync/data/Egypt_Heritage_Artifacts_1000.csv")


# ----------------------------
# MAIN FUNCTION
# ----------------------------

def ask_model(question):
    load_resources()

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
# LOCAL TEST (optional)
# ----------------------------

if __name__ == "__main__":
    print("\nAI Ready. Type 'exit' to quit.\n")

    while True:
        q = input("Ask: ")

        if q.lower() == "exit":
            break

        print(ask_model(q))