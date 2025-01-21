import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from pathlib import Path

from image_classifier.data import AnimalDataModule
from image_classifier.model import ImageClassifier
from google.cloud import storage

def check_gcs_path_exists(gcs_path: str) -> bool:
    """Check if a GCS path exists by trying to access the bucket and prefix."""
    client = storage.Client()
    bucket_name = gcs_path.split('/')[2]  # Extract bucket name from gs://bucket-name/path/to/file
    prefix = '/'.join(gcs_path.split('/')[3:])  # Path inside the bucket

    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0

@hydra.main(config_path="../../configs", config_name="train.yaml")
def main(cfg) -> None:
    model = ImageClassifier(num_classes=10, lr=cfg.hyperparameters.lr)

    # hydra changes working dir to outpurs, getting back to the root
    cur = Path.cwd()
    parent_directory = cur.parent.parent.parent

    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=cfg.hyperparameters.epochs,
        log_every_n_steps=cfg.hyperparameters.log_steps,
        logger=pl.loggers.WandbLogger(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY")),
    )
    
    # if os.path.exists('/gcs/mlops_project25_group72'):
    #     label_folder = 'gs://mlops_project25_group72/data/p'
    #     image_folder = 'gs://mlops_project25_group72/data/p/images'
    # else:
    #     image_folder = str(parent_directory) + '/data/processed/images'
    #     label_folder = str(parent_directory) + '/data/processed
        # Check if the GCS path exists for label and image folders
    label_folder_gcs = '/gcs/mlops_project25_group72/data/p'
    image_folder_gcs = '/gcs/mlops_project25_group72/data/p/images'

    if check_gcs_path_exists(label_folder_gcs) and check_gcs_path_exists(image_folder_gcs):
        # Set the GCS paths as label and image folders directly
        label_folder = label_folder_gcs
        image_folder = image_folder_gcs
    else:
        # Fall back to local path if GCS path doesn't exist (for local testing)
        label_folder = str(parent_directory) + '/data/processed'
        image_folder = str(parent_directory) + '/data/processed/images'

    # Initializing the data module
    data_module = AnimalDataModule(
        Path(label_folder),
        Path(image_folder),
        batch_size=cfg.hyperparameters.batch_size,
        split_ratio=[0.8, 0.1, 0.1],  # 80% train, 10% val, 10% test
        seed=cfg.hyperparameters.seed,
    )

    # Training the model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
