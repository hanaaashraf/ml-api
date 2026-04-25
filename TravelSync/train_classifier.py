import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

MODEL_NAME = "distilbert-base-uncased"

print("Loading classification dataset...")
with open(r"C:\Graduation Project\TravelSync\data\classification_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Encode labels
label_list = df["label"].unique().tolist()
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

df["label"] = df["label"].map(label_to_id)

dataset = Dataset.from_pandas(df)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list)
)

training_args = TrainingArguments(
    output_dir="models/classifier",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_strategy="epoch",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

print("Training classifier...")
trainer.train()

trainer.save_model("models/classifier")
tokenizer.save_pretrained("models/classifier")

# Save label mapping
with open("models/classifier/labels.json", "w", encoding="utf-8") as f:
    json.dump(id_to_label, f)

print("✅ Model trained and saved successfully.")