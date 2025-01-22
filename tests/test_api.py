from fastapi.testclient import TestClient
from image_classifier.api import app
from pathlib import Path

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Animal Classifier API"}

def test_predict_image():
    image_path = Path.cwd() / "data/processed/images/cane_OIP-_3acmW_iSr12XgQTNz0IdQHaFj.jpeg"
    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": image_file},
            headers={"accept": "application/json"}
        )
    assert response.status_code == 200
    assert "predicted_class" in response.json()
    
# run test: pytest tests/test_api.py