import torch
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = r"C:\Graduation Project\models\image_model.pth"

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

# Build model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["classes"]))

model.load_state_dict(checkpoint["model_state"])
model.eval()

classes = checkpoint["classes"]

# Preprocessing (fixed, reusable)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image: Image.Image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]

import pandas as pd

df = pd.read_csv(r"C:\Graduation Project\TravelSync\data\Egypt_Heritage_Artifacts_1000.csv")

def get_info(name):
    row = df[df["Name"] == name.replace("_", " ")]
    if not row.empty:
        return row.iloc[0].to_dict()
    return None