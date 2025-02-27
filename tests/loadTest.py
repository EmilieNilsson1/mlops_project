from locust import HttpUser, between, task
from pathlib import Path
from google.cloud import storage


def check_gcs_path_exists(gcs_path: str) -> bool:
    """Check if a GCS path exists by trying to access the bucket and prefix."""
    storage_client = storage.Client()
    bucket_name = gcs_path.split("/")[
        2
    ]  # Extract bucket name from gs://bucket-name/path/to/file
    prefix = "/".join(gcs_path.split("/")[3:])  # Path inside the bucket

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
    image = (
        Path.cwd() / "data/processed/images/cane_OIP-_3acmW_iSr12XgQTNz0IdQHaFj.jpeg"
    )


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)
    host = "http://127.0.0.1:8000"  # Set the base host for the FastAPI application

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def predict_image(self) -> None:
        if image.startswith("gs://"):
            # Download the image from GCS to a local temporary file
            storage_client = storage.Client()
            bucket_name = image.split("/")[2]
            blob_name = "/".join(image.split("/")[3:])
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            temp_image_path = Path.cwd() / "temp_image.jpeg"
            blob.download_to_filename(temp_image_path)
            image_path = temp_image_path
        else:
            image_path = image

        with open(image_path, "rb") as image_file:
            response = self.client.post(
                "/predict/", files={"file": image_file}, headers={"accept": "text/html"}
            )
        assert response.status_code == 200
        assert "Prediction:" in response.text
        assert "Uploaded Image:" in response.text

        # Clean up the temporary file if it was downloaded from GCS
        if image.startswith("gs://") and temp_image_path.exists():
            temp_image_path.unlink()


# Run the Locust load test: uvicorn src.image_classifier.api:app --reload
# In another terminal: locust -f tests/loadTest.py --host=http://127.0.0.1:8000
