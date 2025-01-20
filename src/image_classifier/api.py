from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from image_classifier.model import ImageClassifier
from image_classifier.data import AnimalDataModule
from pathlib import Path

app = FastAPI()

# Load the model from the checkpoint
parent_directory = Path.cwd()
checkpoint_path = str(parent_directory) + '/outputs/2025-01-17/12-32-13/models/epoch=0-step=328.ckpt' #'/outputs/models/best-checkpoint.ckpt' 
model = ImageClassifier(num_classes=10)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Initialize the data module for preprocessing
label_file = str(parent_directory) + '/data/processed/translated_image_labels.csv'
raw_data_path = str(parent_directory) + '/data/processed/images'
data_module = AnimalDataModule(label_file, raw_data_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image = data_module.train_transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Animal Classifier API"}

# Run the API: uvicorn src.image_classifier.api:app --reload
# Access the API: http://127.0.0.1:8000

# predict picture with curl in new terminal:
# curl -X POST "http://127.0.0.1:8000/predict/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/image.jpg"

