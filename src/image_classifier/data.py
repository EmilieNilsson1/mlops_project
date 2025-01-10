import os
from pathlib import Path
import pandas as pd
import translate
import typer

from torch.utils.data import Dataset

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path, output_folder: Path) -> None:
        self.data_path = raw_data_path
        self.output_folder = output_folder
        self.data = []
        self.df = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        image_name, label = self.data[index]
        return image_name, label

    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print("Preprocessing data...")
        # List all the subfolders inside the dataset folder (each folder should represent an animal)
        class_folders = os.listdir(self.data_path)

        # Iterate over each class folder in the dataset (each folder corresponds to an animal type)
        for class_label in class_folders:
            class_folder = os.path.join(self.data_path, class_label)

            # Skip non-directory files (ensure it's a folder with images)
            if os.path.isdir(class_folder):
                # Iterate over all images inside the class folder
                for image_name in os.listdir(class_folder):
                    if image_name.endswith('.jpg'):  # Assuming image format is JPG
                        image_path = os.path.join(class_folder, image_name)
                        self.data.append([image_name, class_label])  # Append image name and corresponding animal class

        # Create a DataFrame from the collected data
        self.df = pd.DataFrame(self.data, columns=['image_name', 'label'])

        # Translate the 'label' column using the dictionary from translate.py
        self.df['label'] = self.df['label'].map(translate.translate).fillna(self.df['label'])  # Use the dictionary from translate.py

        # Save DataFrame to a CSV file with translated labels
        self.df.to_csv(self.output_folder / 'translated_image_labels.csv', index=False)

        print(f"CSV file saved at '{self.output_folder}/translated_image_labels.csv' with {len(self.df)} entries.")

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """Preprocess the raw data and save it to the output folder."""
    dataset = MyDataset(raw_data_path, output_folder)
    dataset.preprocess()

if __name__ == "__main__":
    typer.run(preprocess)