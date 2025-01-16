import os
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import translate
from shutil import copy2
import typer

class Datahandler(Dataset):
    """Custom dataset compatible with PyTorch Lightning."""
    def __init__(self, processed_data_path: Path, raw_data_path: Path, transform=None):
        self.processed_data_path = processed_data_path
        self.raw_data_path = raw_data_path
        self.transform = transform
        
    def prepare_data(self):
        print("Preprocessing data...")
        data = []
        # Ensure processed_data_path is a Path object and 'images' subdirectory exists
        processed_data_path = Path(self.processed_data_path)
        images_path = processed_data_path / 'images'
        images_path.mkdir(parents=True, exist_ok=True)

        # List all the subfolders inside the dataset folder (each folder should represent an animal)
        class_folders = os.listdir(self.raw_data_path)

        # Iterate over each class folder in the dataset (each folder corresponds to an animal type)
        for class_label in class_folders:
            class_folder = Path(self.raw_data_path) / class_label

            # Skip non-directory files (ensure it's a folder with images)
            if class_folder.is_dir():
                # Iterate over all images inside the class folder
                for image_name in os.listdir(class_folder):
                    # Check if the file is an image and not a hidden system file
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')) and not image_name.startswith('.'):
                        
                        # Construct the full image path
                        image_path = class_folder / image_name
                        
                        # Create a new image name by appending the class label at the end
                        new_image_name = f"{class_label}_{image_name}"
                        
                        # Define the destination path
                        dest_path = images_path / new_image_name
                        
                        # Copy the image to the new destination
                        copy2(image_path, dest_path)
                        
                        # Append the new image name and class label (folder name) to self.data
                        data.append([new_image_name, class_label])

            # Create a DataFrame from the collected data
            self.df = pd.DataFrame(data, columns=['image_name', 'label'])

            # Translate the 'label' column using the dictionary from translate.py
            self.df['label'] = self.df['label'].map(translate.translate).fillna(self.df['label'])

            # Save DataFrame to a CSV file with translated labels
            self.df.to_csv(processed_data_path / 'translated_image_labels.csv', index=False)
            
                   # Corrected path concatenation using /
            csv_path = self.processed_data_path / 'translated_image_labels.csv'
            df = pd.read_csv(csv_path)

            # Create label mappings
            unique_labels = df['label'].unique()
            self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
            
            # Map labels to integers
            df['label'] = df['label'].map(self.label_to_index)
            # data = list(zip(df['image_name'], df['label']))
            

    def _load_labels(self):
        """Load the labels and image names from the provided CSV file."""
        # Corrected path concatenation using /
        csv_path = self.processed_data_path / 'translated_image_labels.csv'
        self.df = pd.read_csv(csv_path)
        
        data = list(zip(self.df['image_name'], self.df['label']))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name, label = self.data[index]
        image_path = self.processed_data_path / 'images' / image_name
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(label)
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
        self.dataset = Datahandler(self.label_file, self.raw_data_path, transform=self.train_transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

def main(processed_data_path: str, raw_data_path: str):
    """Main function to run the image preprocessing."""
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)
    
    datahandler = Datahandler(processed_data_path, raw_data_path)
    datahandler.prepare_data()

if __name__ == "__main__":
    typer.run(main)