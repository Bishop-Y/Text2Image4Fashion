import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mlpt.datamodules.datasets import DeepFashionCaptionDataset
from mlpt.datamodules.datasets import DeepFashionSample
import torch


def collate_deepfashion(batch: list[DeepFashionSample]) -> DeepFashionSample:
    images = torch.stack([item.image for item in batch], dim=0)
    embeddings = torch.stack([item.text_embedding for item in batch], dim=0)
    prompts = [item.prompt for item in batch]
    return DeepFashionSample(image=images,
                             text_embedding=embeddings,
                             prompt=prompts)

class DeepFashionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, workers,
                 max_samples, text_dimension, imsize):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.max_samples = max_samples
        self.text_dimension = text_dimension
        self.imsize = imsize

        self.train_transform = transforms.Compose([
            transforms.Resize((self.imsize, self.imsize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((self.imsize, self.imsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DeepFashionCaptionDataset(
                data_dir=self.data_dir,
                split='train',
                transform=self.train_transform,
                max_samples=self.max_samples,
                text_dimension=self.text_dimension
            )
        if stage == 'test' or stage is None:
            self.test_dataset = DeepFashionCaptionDataset(
                data_dir=self.data_dir,
                split='test',
                transform=self.test_transform,
                max_samples=self.max_samples,
                text_dimension=self.text_dimension
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            drop_last=True,
            collate_fn=collate_deepfashion
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            drop_last=True,
            collate_fn=collate_deepfashion
        )

