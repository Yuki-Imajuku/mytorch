from pathlib import Path
from typing import Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor


class MyCIFAR100(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir: Union[str, Path]
    ) -> None:
        super().__init__()
        # check data_dir
        if isinstance(data_dir, str):
            self.data_dir = Path(data_dir).resolve()
        elif isinstance(data_dir, Path):
            self.data_dir = data_dir.resolve()
        else:
            raise ValueError("data_dir (str | pathlib.Path)")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"DataDir: {str(self.data_dir)}")
        # check
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size (int) > 0")
        self.batch_size = batch_size

    def prepare_data(self):
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    # データセットの作成
    def setup(self, stage=None):
        transform = Compose([
            ToTensor()
        ])
        if stage == "fit" or stage is None:
            full_dataset = CIFAR100(self.data_dir, train=True, transform=transform)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        if stage == "test" or stage is None:
            self.test_dataset = CIFAR100(self.data_dir, train=False, transform=transform)

    # 各データローダーの用意
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    # def teardown(self, stage=None):
    #     pass
