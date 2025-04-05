import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import mlpt.config.config as cfg
from mlpt.datamodules.datasets import DeepFashionCaptionDataset

class DeepFashionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=cfg.TRAIN_BATCH_SIZE, workers=cfg.WORKERS, max_samples=15000):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.max_samples = max_samples
        
        self.train_transform = transforms.Compose([
            transforms.Resize((cfg.IMSIZE, cfg.IMSIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((cfg.IMSIZE, cfg.IMSIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DeepFashionCaptionDataset(
                data_dir=self.data_dir,
                split='train',
                transform=self.train_transform,
                max_samples=self.max_samples
            )
        if stage == 'test' or stage is None:
            self.test_dataset = DeepFashionCaptionDataset(
                data_dir=self.data_dir,
                split='test',
                transform=self.test_transform,
                max_samples=self.max_samples
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            drop_last=True
        )
