import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ORDNA.models.classifier import Classifier
from ORDNA.utils.argparser import get_args, write_config_file
import wandb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import argparse

class MeanRepresentationDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_file: Path, protection_file: Path, habitat_file: Path):
        # Load embeddings
        self.embeddings_df = pd.read_csv(embeddings_file)
        
        # Load protection labels
        protection_df = pd.read_csv(protection_file)
        
        # Load habitat labels
        habitat_df = pd.read_csv(habitat_file)
        
        # Merge all dataframes on 'Sample' or 'spygen_code'
        self.data = self.embeddings_df.merge(protection_df, left_on='Sample', right_on='spygen_code')
        self.data = self.data.merge(habitat_df, on='spygen_code')
        
        # Extract labels
        self.labels = self.data['protection'].values
        
        # One-hot encode habitat
        self.habitat_encoder = OneHotEncoder()
        self.habitat = self.habitat_encoder.fit_transform(self.data[['habitat']]).toarray()
        
        # Extract mean embeddings (assuming the mean columns are named 'Dim1', 'Dim2', etc.)
        embedding_columns = [col for col in self.embeddings_df.columns if col.startswith('Dim')]
        self.embeddings = self.data[embedding_columns].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        habitat = torch.tensor(self.habitat[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample_embedding, habitat, label

# Parsing arguments
parser = argparse.ArgumentParser(description='Training Classifier')
parser.add_argument('--embeddings_file', type=str, required=True, help='Path to the embeddings CSV file')
parser.add_argument('--protection_file', type=str, required=True, help='Path to the protection labels CSV file')
parser.add_argument('--habitat_file', type=str, required=True, help='Path to the habitat labels CSV file')
parser.add_argument('--arg_log', type=bool, default=False, help='Log arguments')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
parser.add_argument('--initial_learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--max_epochs', type=int, default=2, help='Max epochs')
args = parser.parse_args()

# Controllo se la GPU è disponibile
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")

if args.arg_log:
    write_config_file(args)

print("Setting random seed...")
pl.seed_everything(args.seed)

# Percorsi dei file
embeddings_file = Path(args.embeddings_file).resolve()
protection_file = Path(args.protection_file).resolve()
habitat_file = Path(args.habitat_file).resolve()

print("Initializing dataset...")
dataset = MeanRepresentationDataset(embeddings_file=embeddings_file, protection_file=protection_file, habitat_file=habitat_file)

# Suddivisione del dataset in training e validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print("Setting up DataLoaders...")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12)

print("Initializing classifier model...")
# Inizializza il modello del classificatore
sample_emb_dim = dataset.embeddings.shape[1]
habitat_dim = dataset.habitat.shape[1]
model = Classifier(sample_emb_dim=sample_emb_dim, habitat_dim=habitat_dim, num_classes=args.num_classes, initial_learning_rate=args.initial_learning_rate)

print("Setting up checkpoint directory...")
# Checkpoint directory
checkpoint_dir = Path('checkpoints_classifier')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print("Initializing checkpoint callback...")
# General checkpoint callback for best model saving
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    dirpath=checkpoint_dir,
    filename='corse_classifier-{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    mode='max',
)

# Custom Early stopping callback
class CustomEarlyStopping(pl.Callback):
    def __init__(self, monitor: str, patience: int, mode: str = 'min', min_delta: float = 0.0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.wait = 0
        self.best_score = None
        self.stopped_epoch = 0
        self.stop_training = False

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        if self.best_score is None:
            self.best_score = current
            return

        if self.mode == 'min' and current < self.best_score - self.min_delta:
            self.best_score = current
            self.wait = 0
        elif self.mode == 'max' and current > self.best_score + self.min_delta:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True
            self.stop_training = True
            print(f"Early stopping triggered at epoch {self.stopped_epoch} with best {self.monitor}: {self.best_score}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.stop_training:
            print(f"Stopping training at step {trainer.global_step + 1} due to early stopping.")
            trainer.should_stop = True
            trainer.train_loop.run = False  # Stops training immediately

# Callback for validation on each step
class ValidationOnStepCallback(pl.Callback):
    def __init__(self, n_steps):
        super().__init__()
        self.n_steps = n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_step = trainer.global_step + 1
        if current_step % self.n_steps == 0:
            print(f"[DEBUG] Running validation at step {current_step}")
            # Esegui manualmente la validazione
            pl_module.eval()
            val_class_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in trainer.val_dataloaders[0]:
                    sample_embeddings, habitat, labels = batch
                    sample_embeddings, habitat, labels = sample_embeddings.to(pl_module.device), habitat.to(pl_module.device), labels.to(pl_module.device)
                    outputs = pl_module(sample_embeddings, habitat)
                    val_class_loss += pl_module.loss_fn(outputs, labels).item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_class_loss /= len(trainer.val_dataloaders[0])
            val_accuracy = correct / total
            print(f"[DEBUG] Validation at step {current_step}: val_class_loss = {val_class_loss}, val_accuracy = {val_accuracy}")
            pl_module.train()

print("Setting up Wandb logger...")
# Setup logger e trainer
wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir=Path("lightning_logs"), config=args, log_model=False)

# Inizializzazione Wandb
print("Initializing Wandb run...")
wandb_run = wandb.init(project='ORDNA_Class_july', config=args)

# Print Wandb run URL
print(f"Wandb run URL: {wandb_run.url}")

print("Initializing trainer...")

# Parametri del dataset e batch size
N = len(train_dataset)  # Numero di campioni di addestramento
B = args.batch_size  # Batch size

# Calcolare il numero totale di batch per epoca
num_batches_per_epoch = N // B
print(f"Number of batches per epoch: {num_batches_per_epoch}")

# Scegliere n_steps come il 20% dei batch per epoca
n_steps = num_batches_per_epoch // 30
print(f"Validation will run every {n_steps} steps")

# Inizializza i callback
validation_callback = ValidationOnStepCallback(n_steps=n_steps)
early_stopping_callback = CustomEarlyStopping(monitor='val_accuracy', patience=3, mode='max')

trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, validation_callback, early_stopping_callback],
    log_every_n_steps=10,
    detect_anomaly=False
)

# Ensure the callbacks are correctly passed
print(f"Trainer callbacks: {trainer.callbacks}")

# Start training
print("Starting training...")
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Check if early stopping was triggered
if any(callback.stop_training for callback in trainer.callbacks if isinstance(callback, CustomEarlyStopping)):
    stopped_epoch = next(callback.stopped_epoch for callback in trainer.callbacks if isinstance(callback, CustomEarlyStopping))
    print(f"Early stopping was triggered at epoch {stopped_epoch}.")

# Chiudi Wandb
print("Finishing Wandb run...")
wandb.finish()
