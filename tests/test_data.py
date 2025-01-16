import os
import pytest
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from src.image_classifier.data import Datahandler, AnimalDataModule


# Helper function to create mock image data
def create_mock_data(raw_data_path, class_folders, image_count=3):
    for class_name in class_folders:
        class_folder = raw_data_path / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        for i in range(image_count):
            image_path = class_folder / f"image_{i}.jpg"
            image = Image.new("RGB", (100, 100), color=(255, 0, 0))
            image.save(image_path)

@pytest.fixture
def setup_data(tmp_path):
    raw_data_path = tmp_path / "raw_data"
    processed_data_path = tmp_path / "processed_data"

    # Create mock class folders and images
    class_folders = ["cat", "dog", "bird"]
    create_mock_data(raw_data_path, class_folders)

    return raw_data_path, processed_data_path

def test_prepare_data(setup_data):
    raw_data_path, processed_data_path = setup_data
    
    datahandler = Datahandler(processed_data_path, raw_data_path)
    datahandler.prepare_data()

    # Assert the processed data folder and images exist
    images_path = processed_data_path / "images"
    assert images_path.exists()
    assert len(list(images_path.iterdir())) == 9  # 3 classes * 3 images each

    # Assert the CSV file is created
    csv_path = processed_data_path / "translated_image_labels.csv"
    assert csv_path.exists()

    # Validate contents of the CSV
    df = pd.read_csv(csv_path)
    assert len(df) == 9
    assert set(df["label"]) == {"cat", "dog", "chicken"}

def test_dataset_loading(setup_data):
    raw_data_path, processed_data_path = setup_data

    # Prepare data
    datahandler = Datahandler(processed_data_path, raw_data_path)
    datahandler.prepare_data()

    # Initialize dataset and validate loading
    dataset = Datahandler(processed_data_path, raw_data_path, transform=transforms.ToTensor())
    dataset.data = dataset._load_labels()

    assert len(dataset) == 9  # Total number of images

    # Check one sample
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)

def test_animal_data_module(setup_data):
    raw_data_path, processed_data_path = setup_data

    # Prepare data
    datahandler = Datahandler(processed_data_path, raw_data_path)
    datahandler.prepare_data()

    # Initialize AnimalDataModule
    datamodule = AnimalDataModule(processed_data_path / "translated_image_labels.csv", raw_data_path, batch_size=4)
    datamodule.setup()

    # Validate train_dataloader
    train_loader = datamodule.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    batch = next(iter(train_loader))
    images, labels = batch
    assert images.shape[0] == 4  # Batch size
    assert labels.shape[0] == 4
