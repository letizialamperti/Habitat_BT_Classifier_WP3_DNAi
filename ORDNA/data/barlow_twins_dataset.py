import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

class BarlowTwinsDataset(Dataset):
    def __init__(self, embeddings_file: Path, labels_file: Path, habitats_file: Path) -> None:
        self.embeddings_df = pd.read_csv(embeddings_file)
        self.labels_dict = self.load_labels(labels_file)
        self.habitats_dict = self.load_labels(habitats_file)

    def load_labels(self, file_path: Path):
        labels_df = pd.read_csv(file_path)
        return dict(zip(labels_df['spygen_code'], labels_df['label']))

    def __getitem__(self, index: int):
        row = self.embeddings_df.iloc[index]
        sample_name = row['Sample']
        embedding = torch.tensor(row[1:].values, dtype=torch.float)
        label = torch.tensor(self.labels_dict[sample_name], dtype=torch.long)
        habitat = torch.tensor(self.habitats_dict[sample_name], dtype=torch.float)
        return embedding, habitat, label

    def __len__(self) -> int:
        return len(self.embeddings_df)
