from PIL import Image
from predict_image import predict_image
from predict_image import get_info

# Load test image
image = Image.open("test1.jpg")

# Run model
artifact = predict_image(image)

# Get info
info = get_info(artifact)

print("Prediction:", artifact)
print("Info:", info)