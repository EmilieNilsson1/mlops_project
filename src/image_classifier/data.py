import os

from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms
from shutil import copy2
import typer

from image_classifier.translate import translate


class Datahandler(Dataset):
    """Custom dataset compatible with PyTorch Lightning."""

    def __init__(self, processed_data_path: Path, raw_data_path: Path, transform=None):
        self.processed_data_path = processed_data_path
        self.raw_data_path = raw_data_path
        self.transform = transform

        # Initialize self.data as an empty list by default
        self.data = []

        if os.path.exists(self.processed_data_path / "translated_image_labels.csv"):
            self.data = self._load_labels()
        else:
            print(
                "Warning: translated_image_labels.csv not found. Prepare the data first."
            )

    def prepare_data(self):
        print("Preprocessing data...")
        data = []
        processed_data_path = Path(self.processed_data_path)
        images_path = processed_data_path / "images"
        images_path.mkdir(parents=True, exist_ok=True)

        class_folders = os.listdir(self.raw_data_path)
        for class_label in class_folders:
            class_folder = Path(self.raw_data_path) / class_label
            if class_folder.is_dir():
                for image_name in os.listdir(class_folder):
                    if image_name.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    ) and not image_name.startswith("."):
                        image_path = class_folder / image_name
                        new_image_name = f"{class_label}_{image_name}"
                        dest_path = images_path / new_image_name
                        copy2(image_path, dest_path)
                        data.append([new_image_name, class_label])

        self.df = pd.DataFrame(data, columns=["image_name", "label"])
        self.df["label"] = self.df["label"].map(translate).fillna(self.df["label"])
        self.df.to_csv(processed_data_path / "translated_image_labels.csv", index=False)

        # Update self.data after preparing the data
        self.data = list(zip(self.df["image_name"], self.df["label"]))

    def _load_labels(self):
        csv_path = self.processed_data_path / "translated_image_labels.csv"
        self.df = pd.read_csv(csv_path)
        return list(zip(self.df["image_name"], self.df["label"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name, label = self.data[index]
        image_path = self.processed_data_path / "images" / image_name
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(label)


class AnimalDataModule(pl.LightningDataModule):
    """DataModule for PyTorch Lightning."""

    def __init__(
        self,
        label_file: Path,
        raw_data_path: Path,
        batch_size: int = 32,
        split_ratio: list[float] = [0.8, 0.1, 0.1],
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.label_file = label_file
        self.raw_data_path = raw_data_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage=None) -> None:
        """Load datasets for training, validation, and testing."""
        self.dataset = Datahandler(
            self.label_file, self.raw_data_path, transform=self.train_transform
        )

        # Calculate lengths for splits
        total_len = len(self.dataset)
        train_len = int(total_len * self.split_ratio[0])
        val_len = int(total_len * self.split_ratio[1])
        test_len = total_len - train_len - val_len  # Remaining for test

        # Perform the split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_len, val_len, test_len]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def main(processed_data_path: str, raw_data_path: str) -> None:
    """Main function to run the image preprocessing."""
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    datahandler = Datahandler(processed_data_path, raw_data_path)
    datahandler.prepare_data()


if __name__ == "__main__":
    typer.run(main)
