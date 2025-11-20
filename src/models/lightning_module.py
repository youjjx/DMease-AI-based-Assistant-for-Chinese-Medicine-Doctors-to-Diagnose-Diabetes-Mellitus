import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy
from .modules import PaperModel


class ResearchSystem(pl.LightningModule):
    def __init__(self, lr=1e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        self.model = PaperModel(num_classes=num_classes)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }