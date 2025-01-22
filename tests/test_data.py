import shutil
import pytest
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from image_classifier.data import Datahandler, AnimalDataModule

test_dir = Path("tests/mock_data")
raw_data_path = test_dir / "raw"
processed_data_path = test_dir / "processed"


def setup_mock_environment():
    # Create mock directories and files
    (raw_data_path / "cat").mkdir(parents=True, exist_ok=True)
    (raw_data_path / "dog").mkdir(parents=True, exist_ok=True)
    (processed_data_path).mkdir(parents=True, exist_ok=True)

    # Create mock image files
    for i in range(5):
        Image.new("RGB", (200, 200)).save(raw_data_path / "cat" / f"cat_{i}.jpg")
        Image.new("RGB", (200, 200)).save(raw_data_path / "dog" / f"dog_{i}.jpg")


def teardown_mock_environment():
    # Remove mock directories and files
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture(scope="module", autouse=True)
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
    assert set(df["label"]) == {0, 5}  # Validate labels
    assert len(df) == 10  # 5 cat images + 5 dog images


def test_datahandler_load_labels(mock_environment):
    # Preprocess data first
    datahandler = Datahandler(processed_data_path, raw_data_path)
    datahandler.prepare_data()

    # Load labels
    data = datahandler._load_labels()
    assert len(data) == 10  # Validate the number of entries
    for image_name, label in data:
        assert label in [0, 5]  # Validate labels


def test_datahandler_getitem(mock_environment):
    # Preprocess data first
    datahandler = Datahandler(processed_data_path, raw_data_path, transform=None)
    datahandler.prepare_data()

    # Test __getitem__ method
    image, label = datahandler[0]
    assert isinstance(image, Image.Image)  # Check that it returns an image
    assert label in [0, 5]  # Check that label is an integer


def test_animal_data_module(mock_environment):
    # Initialize and setup AnimalDataModule
    data_module = AnimalDataModule(processed_data_path, raw_data_path, batch_size=2)
    data_module.setup()

    # Validate the dataset splits
    assert len(data_module.train_dataset) == 8  # 80% of 10
    assert len(data_module.val_dataset) == 1  # 10% of 10
    assert len(data_module.test_dataset) == 1  # 10% of 10

    # Validate dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    assert len(train_loader) == 4  # Batch size of 2
    assert len(val_loader) == 1  # Single batch
    assert len(test_loader) == 1  # Single batch


def test_animal_data_module_empty_split(mock_environment):
    # Initialize and setup AnimalDataModule with empty split
    data_module = AnimalDataModule(processed_data_path, raw_data_path, batch_size=2, split_ratio=[1.0, 0.0, 0.0])
    data_module.setup()

    # Validate the dataset splits
    assert len(data_module.train_dataset) == 10  # All data goes to training
    assert len(data_module.val_dataset) == 0  # No validation data
    assert len(data_module.test_dataset) == 0  # No test data


def test_datahandler_with_transforms(mock_environment):
    from torchvision import transforms

    # Add a transform
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    # Preprocess data with transforms
    datahandler = Datahandler(processed_data_path, raw_data_path, transform=transform)
    datahandler.prepare_data()

    image, label = datahandler[0]
    assert isinstance(image, torch.Tensor)  # Ensure the image is a tensor
    assert label in [0, 5]  # Check the label


if __name__ == "__main__":
    pytest.main(["-v", __file__])
