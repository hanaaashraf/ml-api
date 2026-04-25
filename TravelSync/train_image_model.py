import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATASET_DIR = r"C:\Graduation Project\TravelSync\dataset_images"
MODEL_PATH = "models/image_model.pth"

# Create models folder
import os
os.makedirs("models", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Classes:", dataset.classes)

for epoch in range(3):
    total_loss = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "classes": dataset.classes
}, MODEL_PATH)

print("✅ Training complete")