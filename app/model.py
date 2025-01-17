import torch.nn as nn
import timm
import pytorch_lightning as pl
import torch

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes: int):
        super(ImageClassifier, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)