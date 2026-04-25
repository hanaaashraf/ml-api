from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import torch
from torchvision import models, transforms
import pandas as pd
import io
import math
import os
import gdown

from TravelSync.ask_model import ask_model

app = FastAPI()

# ================= SAFE CLEANERS =================
def clean_value(x):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return 0.0
    return x

def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_value(v) for k, v in d.items()}
    return d

# ================= IMAGE MODEL =================

MODEL_PATH = "models/image_model.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)

    url = "https://drive.google.com/uc?id=1BSJNf8ca0PB0Uk17uAgKLVo5mYOidL3K"

    gdown.download(url, MODEL_PATH, quiet=False)

    
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model_state"])
model.eval()

classes = checkpoint["classes"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================= DATASET =================
df = pd.read_csv("TravelSync/data/Egypt_Heritage_Artifacts_1000.csv")

# ================= FUNCTIONS =================

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]


def get_info(name):
    cleaned = name.replace("_", " ").strip().lower()

    df_copy = df.copy()
    df_copy["Name"] = df_copy["Name"].astype(str).str.strip().str.lower()

    row = df_copy[df_copy["Name"] == cleaned]

    if not row.empty:
        return row.iloc[0].to_dict()

    return {}

# ================= ROUTES =================

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):
    image_bytes = await file.read()

    artifact = predict_image(image_bytes)
    info = get_info(artifact)

    return {
        "source": "image",
        "prediction": artifact,
        "info": clean_dict(info)
    }

@app.post("/predict-text")
async def predict_text_api(text: str = Form(...)):
    result = ask_model(text)

    return {
        "source": "text",
        "result": clean_dict(result) if isinstance(result, dict) else result
    }

@app.post("/recommend")
async def recommend(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    if file:
        image_bytes = await file.read()
        artifact = predict_image(image_bytes)
        info = get_info(artifact)

        return {
            "source": "image",
            "prediction": artifact,
            "info": clean_dict(info)
        }

    if text:
        result = ask_model(text)

        return {
            "source": "text",
            "result": clean_dict(result) if isinstance(result, dict) else result
        }

    return {"error": "Provide text or image"}