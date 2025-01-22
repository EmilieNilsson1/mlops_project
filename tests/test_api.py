from fastapi.testclient import TestClient
from image_classifier.api import app
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account
import os
import json

client = TestClient(app)

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    credentials_json = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    service_account.Credentials.from_service_account_info(json.loads(credentials_json))


def check_gcs_path_exists(gcs_path: str) -> bool:
    """Check if a GCS path exists by trying to access the bucket and prefix."""
    storage_client = storage.Client()
    bucket_name = gcs_path.split('/')[2]  # Extract bucket name from gs://bucket-name/path/to/file
    prefix = '/'.join(gcs_path.split('/')[3:])  # Path inside the bucket

    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0

# Check if the GCS paths exist, if not fall back to local paths
# Get the GCS paths for the label and image folders
image_gcs = "gs://mlops_project25_group72/data/p/images/cane_OIP-_3acmW_iSr12XgQTNz0IdQHaFj.jpeg"

if check_gcs_path_exists(image_gcs):
    # Set the GCS paths as label and image folders directly
    image = image_gcs
else:
    # Fall back to local path if GCS path doesn't exist (for local testing)
    image = Path.cwd() / "data/processed/images/cane_OIP-_3acmW_iSr12XgQTNz0IdQHaFj.jpeg"

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Animal Classifier API"}

def test_predict_image():
    if image.startswith("gs://"):
        # Download the image from GCS to a local temporary file
        storage_client = storage.Client()
        bucket_name = image.split('/')[2]
        blob_name = '/'.join(image.split('/')[3:])
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        temp_image_path = Path.cwd() / "temp_image.jpeg"
        blob.download_to_filename(temp_image_path)
        image_path = temp_image_path
    else:
        image_path = image

    with open(image_path, "rb") as image_file:
        response = client.post("/predict/", files={"file": image_file}, headers={"accept": "application/json"})
    assert response.status_code == 200
    assert "predicted_class" in response.json()

    # Clean up the temporary file if it was downloaded from GCS
    if image.startswith("gs://") and temp_image_path.exists():
        temp_image_path.unlink()

# run test: pytest tests/test_api.py