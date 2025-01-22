from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from image_classifier.model import ImageClassifier
from image_classifier.data import AnimalDataModule
from image_classifier.translate import translate
from pathlib import Path
from google.cloud import storage
from io import BytesIO
import requests
from fastapi import Form
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the static files directory
app.mount("/data/static", StaticFiles(directory="data/static"), name="static")

static_dir = 'data/static'
os.makedirs(static_dir, exist_ok=True)

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Save the uploaded image to the static folder
        image_path = f"data/static/{file.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = data_module.train_transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

            # Translate the predicted class index to a label
            translated_class = translate[predicted_class]  # Access the dictionary with the index

        # Generate image URL
        image_url = f"/{image_path}"  # Ensure the image is served from static folder

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": translated_class,
            "image_url": image_url,  # Pass the image URL back to the template
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
        })


# Run the API: uvicorn src.image_classifier.api:app --reload

# Access the API: http://127.0.0.1:8000

# predict picture with curl in new terminal:
# curl -X POST "http://127.0.0.1:8000/predict/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/image.jpg"