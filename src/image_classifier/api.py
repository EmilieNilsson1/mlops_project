from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import sys
from image_classifier.model import ImageClassifier
from image_classifier.data import AnimalDataModule
from image_classifier.translate import translate
from google.cloud import storage
from io import BytesIO
from pathlib import Path
import os
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Create the static folder
os.makedirs("data/static", exist_ok=True)

# Mount the static files directory
app.mount("/data/static", StaticFiles(directory="data/static"), name="static")

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

label_file = "/gcs/mlops_project25_group72/data/p/"
raw_data_path = "/gcs/mlops_project25_group72/data/p/images"
data_module = AnimalDataModule(label_file, raw_data_path)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Save the uploaded image to the local static folder
        image_data = file.file.read()
        image_name = file.filename
        local_image_path = Path("data/static") / image_name
        with open(local_image_path, "wb") as buffer:
            buffer.write(image_data)

        # Generate the local image URL
        local_image_url = f"/data/static/{image_name}"

        # Open and preprocess the image
        image = Image.open(local_image_path).convert("RGB")
        image_tensor = data_module.train_transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            translated_class = translate[predicted_class]

        if not "pytest" in sys.modules:
            # Save the image to GCS with the prediction number and timestamp in the name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gcs_image_name = f"{predicted_class}_{timestamp}.jpeg"
            blob = bucket.blob(f"data/preds/{gcs_image_name}")
            blob.upload_from_filename(local_image_path)

        # Generate the GCS image URL
        gcs_image_url = (
            f"https://storage.googleapis.com/{bucket_name}/data/preds/{gcs_image_name}"
        )

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": translated_class,
                "image_url": local_image_url,  # Pass the local image URL back to the template
                "gcs_image_url": gcs_image_url,  # Pass the GCS image URL back to the template
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(e),
            },
        )


# Run the API: uvicorn src.image_classifier.api:app --reload

# Access the API: http://127.0.0.1:8000