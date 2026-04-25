from PIL import Image
import os
from PIL import Image

DATASET_DIR = "dataset_images"

deleted = 0

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        path = os.path.join(root, file)

        try:
            img = Image.open(path).convert("RGB")
            img.save(path)
        except:
            os.remove(path)
            deleted += 1

print(f"Deleted {deleted} corrupted images")