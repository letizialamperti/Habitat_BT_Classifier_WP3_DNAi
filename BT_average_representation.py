import torch
import pandas as pd
import numpy as np
import os
import csv
from pathlib import Path
from IPython.display import display, clear_output
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.sequence_mapper import SequenceMapper

MODEL_TYPE = 'barlow_twins'  # Alternativa: 'triplets'
CHECKPOINT_PATH = Path('checkpoints/BT_cote_france_dataset-epoch=00.ckpt')
DATASET = 'dataset_cote_france'
SAMPLE_DIR = Path(f'/store/sdsc/sd29/letizia/cote_france')
SEQUENCE_LENGTH = 300
SAMPLE_SUBSET_SIZE = 500
NUM_CLASSES = 4  # Adjust this based on the number of classes in your classifier

# Caricamento del modello
if MODEL_TYPE == 'barlow_twins':
    model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(CHECKPOINT_PATH)
else:
    raise Exception('Unknown model type:', MODEL_TYPE)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Preparazione dei file per l'output
version = CHECKPOINT_PATH.parents[1].name
output_folder = "/scratch/snx3000/llampert/embedding_coords/"
os.makedirs(output_folder, exist_ok=True)
output_csv_file = os.path.join(output_folder, f"new_embedding_coordinates_{DATASET.lower()}_{version}_.csv")

# Mapper per le sequenze
sequence_mapper = SequenceMapper()

# Processamento dei file
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sample"] + [f"Dim{i+1}" for i in range(model.hparams.sample_emb_dim)] + [f"Standard_dev_{i+1}" for i in range(model.hparams.sample_emb_dim)])

    num_files = len(list(SAMPLE_DIR.rglob('*.csv')))
    for i, file_path in enumerate(SAMPLE_DIR.rglob('*.csv')):
        sample_name = file_path.stem
        sample_df = pd.read_csv(file_path)
        sample_df = sample_df.sample(frac=1)  # Random shuffle

        sample_emb_coords = []
        for start in range(0, len(sample_df), SAMPLE_SUBSET_SIZE):
            end = min(start + SAMPLE_SUBSET_SIZE, len(sample_df))
            batch_df = sample_df.iloc[start:end]
            forward_sequences = batch_df['Forward'].tolist()
            reverse_sequences = batch_df['Reverse'].tolist()

            # Convert sequence data to tensor
            forward_tensor = torch.tensor(sequence_mapper.map_seq_list(forward_sequences, SEQUENCE_LENGTH))
            reverse_tensor = torch.tensor(sequence_mapper.map_seq_list(reverse_sequences, SEQUENCE_LENGTH))
            input_tensor = torch.stack((forward_tensor, reverse_tensor), dim=1).to(device)
            input_tensor = input_tensor.unsqueeze(0)  # Adjusting the batch dimension

            with torch.no_grad():
                sample_emb = model(input_tensor)  # Get the embeddings
                sample_emb_coords.append(sample_emb.squeeze().cpu().numpy())

        # Ensure sample_emb_coords is not empty
        if len(sample_emb_coords) > 0:
            # Calculate the mean and standard deviation of the embeddings
            sample_emb_coords = np.array(sample_emb_coords)
            mean_coords = np.mean(sample_emb_coords, axis=0)
            std_coords = np.std(sample_emb_coords, axis=0)
        else:
            mean_coords = np.zeros(model.hparams.sample_emb_dim)
            std_coords = np.zeros(model.hparams.sample_emb_dim)

        print(f"DEBUG - mean_coords: {mean_coords}")
        print(f"DEBUG - std_coords: {std_coords}")

        # Write results to CSV
        writer.writerow([sample_name] + mean_coords.tolist() + std_coords.tolist())
        display(f'Processed {i+1}/{num_files} files')
        clear_output(wait=True)

print(f"File CSV creato con successo: {output_csv_file}")
