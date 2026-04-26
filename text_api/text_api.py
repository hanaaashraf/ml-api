from fastapi import FastAPI, Form
import os
import gdown

from ask_model import ask_model

app = FastAPI()

MODEL_PATH = "models/classifier"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading text model...")

        os.makedirs("models", exist_ok=True)

        gdown.download_folder(
            "https://drive.google.com/drive/folders/1VLboJGUbPsoJf7zNZuxn0d3CK_8dSoQc?usp=drive_link",
            output=MODEL_PATH,
            quiet=False
        )

@app.get("/")
def home():
    return {"status": "text api running"}

@app.post("/predict-text")
async def predict_text(text: str = Form(...)):
    ensure_model()

    result = ask_model(text)

    return {"result": result}