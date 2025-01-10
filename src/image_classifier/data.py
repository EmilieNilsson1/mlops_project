import os
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import typer

class MyLightningDataset(Dataset):
    """Custom dataset compatible with PyTorch Lightning."""

    def __init__(self, label_file: Path, raw_data_path: Path, transform=None):
        self.label_file = label_file
        self.raw_data_path = raw_data_path
        self.transform = transform
        self.data = self._load_labels()

    def _load_labels(self):
        """Load the labels and image names from the provided CSV file."""
        df = pd.read_csv(self.label_file)
        # Create label mappings
        unique_labels = df['label'].unique()
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        
        # Map labels to integers
        df['label'] = df['label'].map(self.label_to_index)
        data = list(zip(df['image_name'], df['label']))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name, label = self.data[index]
        image_path = os.path.join(self.raw_data_path, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(label)  # Ensure label is an integer

class AnimalDataModule(pl.LightningDataModule):
    """DataModule for PyTorch Lightning."""

    def __init__(self, label_file: Path, raw_data_path: Path, batch_size: int = 32):
        super().__init__()
        self.label_file = label_file
        self.raw_data_path = raw_data_path
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        """Load datasets for training, validation, and testing."""
        self.dataset = MyLightningDataset(self.label_file, self.raw_data_path, transform=self.train_transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
