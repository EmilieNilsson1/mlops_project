import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from pathlib import Path

from image_classifier.data import AnimalDataModule
from image_classifier.model import ImageClassifier


@hydra.main(config_path="../../configs", config_name="train.yaml")
def main(cfg) -> None:
    model = ImageClassifier(num_classes=10, lr=cfg.hyperparameters.lr)

    # hydra changes working dir to outpurs, getting back to the root
    cur = Path.cwd()
    parent_directory = cur.parent.parent.parent

    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=cfg.hyperparameters.epochs,
        log_every_n_steps=cfg.hyperparameters.log_steps,
        logger=pl.loggers.WandbLogger(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY")),
        callbacks=[checkpoint_callback],
        max_epochs=cfg.hyperparameters.epochs,
        log_every_n_steps=cfg.hyperparameters.log_steps,
        logger=pl.loggers.WandbLogger(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY")),
    )

    # Initializing the data module
    data_module = AnimalDataModule(
        Path(str(parent_directory) + "/data/processed"),
        Path(str(parent_directory) + "/data/processed/images"),
        Path(str(parent_directory) + "/data/processed"),
        Path(str(parent_directory) + "/data/processed/images"),
        batch_size=cfg.hyperparameters.batch_size,
        split_ratio=[0.8, 0.1, 0.1],  # 80% train, 10% val, 10% test
        seed=cfg.hyperparameters.seed,
    )

    # Training the model
    trainer.fit(model, data_module)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
