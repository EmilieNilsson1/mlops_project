import pytorch_lightning as pl
import typer
from pathlib import Path

from image_classifier.model import ImageClassifier
from image_classifier.data import AnimalDataModule

app = typer.Typer()


@app.command()
def main(
    ckpt: str = typer.Argument(...),
    label_file: Path = typer.Argument(...),
    data_path: str = typer.Argument(...),
) -> None:
    model = ImageClassifier.load_from_checkpoint(ckpt, num_classes=10)
    model.eval()
    trainer = pl.Trainer()
    trainer.test(model, datamodule=AnimalDataModule(label_file, data_path))


if __name__ == "__main__":
    app()
