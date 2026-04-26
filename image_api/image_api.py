from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import models, transforms
import io
import os
import gdown

app = FastAPI()

MODEL_PATH = "models/image_model.pth"

model = None
classes = None

def load_model():
    global model, classes

    if model is None:
        print("Loading image model...")

        if not os.path.exists(MODEL_PATH):
            os.makedirs("models", exist_ok=True)

            gdown.download(
                "https://drive.google.com/uc?id=1BSJNf8ca0PB0Uk17uAgKLVo5mYOidL3K",
                MODEL_PATH,
                quiet=False
            )

        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(
            model.fc.in_features,
            len(checkpoint["classes"])
        )

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        classes = checkpoint["classes"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image_bytes):
    load_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]


@app.get("/")
def home():
    return {"status": "image api running"}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    result = predict(image_bytes)

    return {"prediction": result}