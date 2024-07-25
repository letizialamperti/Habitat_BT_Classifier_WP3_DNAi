import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import pytorch_lightning as pl

class HabitatDataModule(pl.LightningDataModule):
    def __init__(self, labels_file, batch_size):
        super().__init__()
        self.labels_file = labels_file
        self.batch_size = batch_size
        self.one_hot_encoder = OneHotEncoder()
        self.habitats = None

    def setup(self, stage=None):
        labels = pd.read_csv(self.labels_file)
        self.habitats = self.one_hot_encoder.fit_transform(labels[['habitat']]).toarray()
        self.habitats = torch.tensor(self.habitats, dtype=torch.float)

    def train_dataloader(self):
        return DataLoader(HabitatDataset(self.habitats), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(HabitatDataset(self.habitats), batch_size=self.batch_size)

class HabitatDataset(Dataset):
    def __init__(self, habitats):
        self.habitats = habitats

    def __len__(self):
        return len(self.habitats)

    def __getitem__(self, idx):
        return self.habitats[idx]
