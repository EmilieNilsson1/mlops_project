import os
import shutil
import pytest
import pandas as pd
from pathlib import Path
from PIL import Image
from src.image_classifier.data import Datahandler, AnimalDataModule
from unittest.mock import patch, MagicMock

test_dir = Path("tests/mock_data")
raw_data_path = test_dir / "raw"
processed_data_path = test_dir / "processed"

def setup_mock_environment():
    # Create mock directories and files
    (raw_data_path / "cat").mkdir(parents=True, exist_ok=True)
    (raw_data_path / "dog").mkdir(parents=True, exist_ok=True)

    # Create mock image files
    for i in range(3):
        Image.new('RGB', (100, 100)).save(raw_data_path / "cat" / f"cat_{i}.jpg")
        Image.new('RGB', (100, 100)).save(raw_data_path / "dog" / f"dog_{i}.jpg")

def teardown_mock_environment():
    # Remove mock directories and files
    if test_dir.exists():
        shutil.rmtree(test_dir)

@pytest.fixture(scope="function")
def mock_environment():
    setup_mock_environment()
    yield
    teardown_mock_environment()

def test_datahandler_prepare_data(mock_environment):
    datahandler = Datahandler(processed_data_path, raw_data_path)
    datahandler.prepare_data()

    # Check if processed images directory and CSV file were created
    images_path = processed_data_path / "images"
    print(images_path)
    assert images_path.exists()
    assert (processed_data_path / "translated_image_labels.csv").exists()

    # Validate the contents of the CSV file
    df = pd.read_csv(processed_data_path / "translated_image_labels.csv")
    assert set(df["label"]) == {"cat", "dog"}  # Validate labels
    assert len(df) == 6  # 3 cat images + 3 dog images

def test_datahandler_load_labels(mock_environment):
    # Preprocess data first
    datahandler = Datahandler(processed_data_path, raw_data_path)
    datahandler.prepare_data()

    # Load labels
    data = datahandler._load_labels()
    assert len(data) == 6  # Validate the number of entries
    for image_name, label in data:
        assert label in ["cat", "dog"]  # Validate labels

def test_datahandler_getitem(mock_environment):
    # Preprocess data first
    datahandler = Datahandler(processed_data_path, raw_data_path, transform=None)
    datahandler.prepare_data()

    # Test __getitem__ method
    image, label = datahandler[0]
    assert isinstance(image, Image.Image)  # Check that it returns an image
    assert label in [0, 1]  # Check that label is an integer

def test_animal_data_module(mock_environment):
    # Initialize and setup AnimalDataModule
    data_module = AnimalDataModule(processed_data_path, raw_data_path, batch_size=2)
    data_module.setup()

    # Validate the dataset splits
    assert len(data_module.train_dataset) == 4  # 80% of 6
    assert len(data_module.val_dataset) == 1  # 10% of 6
    assert len(data_module.test_dataset) == 1  # 10% of 6

    # Validate dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    assert len(train_loader) == 2  # Batch size of 2
    assert len(val_loader) == 1  # Single batch
    assert len(test_loader) == 1  # Single batch

if __name__ == "__main__":
    pytest.main(["-v", __file__])
