import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from pathlib import Path
from google.cloud import storage
import typer
from typing_extensions import Annotated
from typing import Optional

from image_classifier.data import AnimalDataModule
from image_classifier.model import ImageClassifier


app = typer.Typer()


def check_gcs_path_exists(gcs_path: str) -> bool:
    """Check if a GCS path exists by trying to access the bucket and prefix."""
    client = storage.Client()
    bucket_name = gcs_path.split("/")[
        2
    ]  # Extract bucket name from gs://bucket-name/path/to/file
    prefix = "/".join(gcs_path.split("/")[3:])  # Path inside the bucket

    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0


@app.command()
def main(env: Annotated[str, typer.Option("--run", "-o")] = "cloud"):
    """Wrapper funtion for train, to spedicify the running environment
    to run this locally :
        python src/image_classifier/train.py --run local
    to run on cloud:
        python src/image_classifier/train.py OR train
    """
    os.environ["RUNNING_ENV"] = env
    print(f"Running in {os.getenv('RUNNING_ENV')} mode")
    sys.argv = [arg for arg in sys.argv if arg != "--run" and arg != "local"]
    _train()


@hydra.main(config_path="../../configs", config_name="train.yaml")
def _train(cfg) -> None:

    env = os.getenv("RUNNING_ENV")
    print(f"Running in {env} mode")

    # Set seed for reporducability
    pl.seed_everything(cfg.hyperparameters.seed, workers=True)

    model = ImageClassifier(num_classes=10, lr=cfg.hyperparameters.lr)

    # hydra changes working dir to outpurs, getting back to the root
    cur = Path.cwd()
    parent_directory = cur.parent.parent.parent

    # checkpoint_callback = ModelCheckpoint(dirpath="/gcs/mlops_project25_group72/models", monitor="val_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(
        dirpath="gs://mlops_project25_group72/models", monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=cfg.hyperparameters.epochs,
        log_every_n_steps=cfg.hyperparameters.log_steps,
        logger=pl.loggers.WandbLogger(
            project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"),
            config=dict(cfg)
        ),
    )
    # Check if the GCS paths exist, if not fall back to local paths
    # Get the GCS paths for the label and image folders
    label_folder_gcs = "/gcs/mlops_project25_group72/data/p"
    image_folder_gcs = "/gcs/mlops_project25_group72/data/p/images"

    if (
        check_gcs_path_exists(label_folder_gcs)
        and check_gcs_path_exists(image_folder_gcs)
        and env == "cloud"
    ):
        # Set the GCS paths as label and image folders directly
        label_folder = label_folder_gcs
        image_folder = image_folder_gcs
    else:
        # Fall back to local path if GCS path doesn't exist (for local testing)
        label_folder = str(parent_directory) + "/data/processed"
        image_folder = str(parent_directory) + "/data/processed/images"

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
    # to run this locally :
    # python src/image_classifier/train.py --run local
    # to run on cloud:
    # python src/image_classifier/train.py OR train
    app()
