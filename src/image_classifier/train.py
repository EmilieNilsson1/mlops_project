import os

import pytorch_lightning as pl
import wandb

from data import AnimalDataModule
from model import ImageClassifier

def main():
    # TODO: hyperparameters from config file here

    model = ImageClassifier(num_classes=10)

    trainer = pl.Trainer(
                        max_epochs=10,
                        log_every_n_steps=10
                        #logger=pl.loggers.WandbLogger(project=os.environ['WANDB_PROJECT'], entity=os.environ['WANDB_ENTITY'])
    )

    trainer.fit(model, 
                AnimalDataModule('data/processed/translated_image_labels.csv', 'data/processed/images'))

if __name__ == "__main__":
    main()