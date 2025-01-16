import os

import pytorch_lightning as pl
import wandb
import hydra
from pathlib import Path

from data import AnimalDataModule
from model import ImageClassifier

@hydra.main(config_path='../../configs', config_name='train')
def main(cfg):
    # TODO: hyperparameters from config file here

    model = ImageClassifier(num_classes=10)

    trainer = pl.Trainer(
                        max_epochs=cfg.hyperparameters.epochs,
                        log_every_n_steps=cfg.hyperparameters.log_steps,
                        logger=pl.loggers.WandbLogger(project=os.getenv('WANDB_PROJECT'), entity=os.getenv('WANDB_ENTITY'))
    )

    # hydra changes working dir to outpurs, getting back to the root
    cur = Path.cwd()
    parent_directory = cur.parent.parent.parent
    
    trainer.fit(model, 
                AnimalDataModule(str(parent_directory) + '/data/processed', str(parent_directory) + '/data/processed/images'))

if __name__ == "__main__":
    main()