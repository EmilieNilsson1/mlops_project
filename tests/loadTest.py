from locust import HttpUser, between, task
from pathlib import Path


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
        """A task that simulates a user uploading an image to the predict endpoint."""
        image_path = Path.cwd() / "data/processed/images/cane_OIP-_3acmW_iSr12XgQTNz0IdQHaFj.jpeg"
        with open(image_path, "rb") as image_file:
            self.client.post("/predict/", files={"file": image_file}, headers={"accept": "application/json"})


# Run the Locust load test: uvicorn src.image_classifier.api:app --reload
# In another terminal: locust -f tests/test_api.py --host=http://127.0.0.1:8000
