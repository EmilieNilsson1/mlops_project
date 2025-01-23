import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl


class ImageClassifier(pl.LightningModule):
    """
    A simple image classifier using a pretrained ResNet18 model.
    """

    def __init__(self, num_classes: int, lr: float = 1e-3) -> None:
        super(ImageClassifier, self).__init__()
        self.lr = lr
        self.model = timm.create_model("resnet18", pretrained=True)

        # remove final layer and replace with linear layer with num_classes outputs
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        im, label = batch
        pred = self(im)
        loss = self.criterion(pred, label)
        acc = (pred.argmax(1) == label).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        im, label = batch
        pred = self(im)
        loss = self.criterion(pred, label)
        acc = (pred.argmax(1) == label).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        im, label = batch
        pred = self(im)
        loss = self.criterion(pred, label)
        acc = (pred.argmax(1) == label).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
