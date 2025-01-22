from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from image_classifier.model import ImageClassifier
from image_classifier.data import AnimalDataModule
from pathlib import Path
from google.cloud import storage
from io import BytesIO

app = FastAPI()

# Define GCS bucket details
bucket_name = "mlops_project25_group72"
blob_name = "models/epoch=0-step=328.ckpt"

# Initialize GCS client
client = storage.Client()

# Fetch the blob from the bucket
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(blob_name)
checkpoint_data = blob.download_as_bytes()

# Load the checkpoint on the desired device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(BytesIO(checkpoint_data), map_location=device)

# Initialize and load the model
model = ImageClassifier(num_classes=10)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

label_file = '/gcs/mlops_project25_group72/data/p/'
raw_data_path = '/gcs/mlops_project25_group72/data/p/images'
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
