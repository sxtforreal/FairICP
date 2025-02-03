import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import config
import pandas as pd
import torchvision.transforms as transforms


class icmDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        accession = data_row.Accession_Number
        video_dir = data_row.video_path
        disease = data_row.ICM

        # Preprocess video frames
        frames = np.load(video_dir)
        frames = np.repeat(frames[..., np.newaxis], 3, axis=-1)
        frames = torch.from_numpy(frames).float() / 255
        frames = frames.permute(2, 3, 0, 1)

        return [
            accession,
            torch.tensor(disease),
            frames,
        ]


def collate_func(batch):
    (
        accession_list,
        disease_list,
        video_list,
    ) = zip(*batch)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Stack list of tensors
    disease_label_tsr = torch.stack(disease_list, dim=0)
    video_tsr = torch.stack(video_list, dim=0)
    # Normalize videos
    B, T, C, H, W = video_tsr.shape
    video_tsr = video_tsr.view(B * T, C, H, W)
    video_tsr = normalize(video_tsr)
    video_tsr = video_tsr.view(B, T, C, H, W)

    return {
        "accessions": list(accession_list),
        "labels": disease_label_tsr,
        "videos": video_tsr,
    }


def create_dataloader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_func,
    )


class icmDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def setup(self, stage=None):
        self.train_dataset = icmDataset(self.train_df)
        self.val_dataset = icmDataset(self.val_df)
        self.test_dataset = icmDataset(self.test_df)

    def train_dataloader(self):
        return create_dataloader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return create_dataloader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return create_dataloader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )


if __name__ == "__main__":
    data_dir = "" # your dir here
    train_df = pd.read_csv(data_dir + "train_df.csv")
    val_df = pd.read_csv(data_dir + "val_df.csv")
    test_df = pd.read_csv(data_dir + "test_df.csv")
    data_module = icmDataModule(train_df, val_df, test_df)
    data_module.setup()

    train = data_module.train_dataloader()
    for batch in train:
        b = batch
        break
