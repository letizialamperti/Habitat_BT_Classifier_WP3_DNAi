import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class MergedDataset(Dataset):
    def __init__(self, embeddings_file: str, protection_file: str, habitat_file: str):
        self.embeddings = pd.read_csv(embeddings_file)
        self.protection = pd.read_csv(protection_file)
        self.habitat = pd.read_csv(habitat_file)

        # Merge datasets on the 'Sample' column
        self.data = pd.merge(self.embeddings, self.protection, left_on='Sample', right_on='spygen_code')
        self.data = pd.merge(self.data, self.habitat, on='spygen_code')

        # One-hot encode habitat
        self.data = pd.get_dummies(self.data, columns=['habitat'], prefix='', prefix_sep='')

        # Extract embeddings
        self.embeddings = self.data.iloc[:, 1:self.data.columns.get_loc('protection')].values
        # Extract one-hot encoded habitat
        habitat_start_index = self.data.columns.get_loc('protection') + 1
        self.habitats = self.data.iloc[:, habitat_start_index:].values
        # Extract protection labels
        self.labels = self.data['protection'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float)
        habitat = torch.tensor(self.habitats[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, habitat, label

class MergedDataModule(pl.LightningDataModule):
    def __init__(self, embeddings_file: str, protection_file: str, habitat_file: str, batch_size: int):
        super().__init__()
        self.embeddings_file = embeddings_file
        self.protection_file = protection_file
        self.habitat_file = habitat_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = MergedDataset(self.embeddings_file, self.protection_file, self.habitat_file)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
