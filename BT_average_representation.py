import torch
import pandas as pd
import numpy as np
import os
import csv
from pathlib import Path
from IPython.display import display, clear_output
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.sequence_mapper import SequenceMapper



CHECKPOINT_PATH = Path('checkpoints/BT_sud_cose_1_dataset-epoch=00.ckpt')  # Percorso del checkpoint
DATASET = 'sud_corse'  # Nome del dataset
SAMPLE_DIR = Path(f'/store/sdsc/sd29/letizia/sud_corse')  # Percorso della directory dei campioni
SEQUENCE_LENGTH = 300  # Lunghezza delle sequenze
SAMPLE_SUBSET_SIZE = 500  # Dimensione del subset del campione
NUM_CLASSES = 2  # Numero di classi
repr_dim = 64  # Dimensione della rappresentazione (specificare correttamente)


# Carica il modello Barlow Twins
model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(CHECKPOINT_PATH, num_classes=NUM_CLASSES)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

version = CHECKPOINT_PATH.parents[1].name
output_folder = "/scratch/snx3000/llampert/embedding_coords/"
os.makedirs(output_folder, exist_ok=True)
output_csv_file = os.path.join(output_folder, f"embedding_coordinates_{DATASET.lower()}_{version}.csv")

sequence_mapper = SequenceMapper()

with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ["Sample"] + [f"Dim{i+1}" for i in range(model.repr_dim)]
    writer.writerow(header)

    num_files = len(list(SAMPLE_DIR.rglob('*.csv')))
    for i, file_path in enumerate(SAMPLE_DIR.rglob('*.csv')):
        sample_name = file_path.stem
        sample_df = pd.read_csv(file_path)
        sample_df = sample_df.sample(frac=1)

        sample_emb_coords = []
        for start in range(0, len(sample_df), SAMPLE_SUBSET_SIZE):
            end = min(start + SAMPLE_SUBSET_SIZE, len(sample_df))
            batch_df = sample_df.iloc[start:end]
            forward_sequences = batch_df['Forward'].tolist()
            reverse_sequences = batch_df['Reverse'].tolist()

            forward_tensor = torch.tensor(sequence_mapper.map_seq_list(forward_sequences, SEQUENCE_LENGTH))
            reverse_tensor = torch.tensor(sequence_mapper.map_seq_list(reverse_sequences, SEQUENCE_LENGTH))
            input_tensor = torch.stack((forward_tensor, reverse_tensor), dim=1).to(device)
            input_tensor = input_tensor.unsqueeze(0)

            with torch.no_grad():
                sample_emb, _ = model(input_tensor)
                sample_emb_coords.append(sample_emb.squeeze().cpu().numpy())

        if len(sample_emb_coords) > 0:
            sample_emb_coords = np.array(sample_emb_coords)
            mean_coords = np.mean(sample_emb_coords, axis=0)
        else:
            mean_coords = np.zeros(model.repr_dim)

        row = [sample_name] + list(mean_coords)
        writer.writerow(row)
        display(f'Processed {i+1}/{num_files} files')
        clear_output(wait=True)

print(f"File CSV creato con successo: {output_csv_file}")
