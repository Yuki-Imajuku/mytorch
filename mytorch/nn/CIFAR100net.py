from pytorch_lightning import LightningModule
import torch

from ..metrics import get_full_metrics


class CIFAR100Net(LightningModule):
    def __init__(self, hidden_size=1000, hidden_channels=16, lr=0.001):
        super().__init__()
        # クラス変数の定義
        self.lr = lr
        self.input_size = 32 * 32
        self.output_size = 100
        self.train_metrics = get_full_metrics(num_classes=self.output_size, prefix="train_")
        self.val_metrics = get_full_metrics(num_classes=self.output_size, prefix="val_")
        self.test_metrics = get_full_metrics(num_classes=self.output_size, prefix="test_")

        # ネットワークの定義
        self.activation = torch.nn.ReLU()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=3, padding=1),
            self.activation,
            torch.nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            self.activation,
            torch.nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=3, padding=1)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size),
            self.activation,
            torch.nn.Linear(hidden_size, self.output_size)
        )
        self.flatten = torch.nn.Flatten()
        self.output = torch.nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.output(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        self.train_metrics(preds, targets)
        # log
        self.log("train_loss", loss)
        self.log_dict(self.train_metrics, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        self.val_metrics(preds, targets)
        # log
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        self.test_metrics(preds, targets)
        # log
        self.log("test_loss", loss, prog_bar=True)
        self.log_dict(self.test_metrics, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
