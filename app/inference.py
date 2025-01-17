from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from model import ImageClassifier
from image_processing import transform_image

router = APIRouter()

# Load the model
model = ImageClassifier(num_classes=10)
model.load_state_dict(torch.load("models/image_classifier.pth"))
model.eval()

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image = transform_image(image)
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)