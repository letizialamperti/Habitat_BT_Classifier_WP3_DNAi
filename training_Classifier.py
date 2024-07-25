import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from merged_dataset import MergedDataModule
from classifier import Classifier
from utils.argparser import get_args, write_config_file
import wandb

# Funzione per calcolare i pesi delle classi
def calculate_class_weights(labels_file: Path, num_classes: int) -> torch.Tensor:
    import pandas as pd
    labels = pd.read_csv(labels_file)
    label_counts = labels['protection'].value_counts().sort_index()
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    return torch.tensor(class_weights.values, dtype=torch.float)

# Controllo se la GPU è disponibile
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")

# Usa la stessa configurazione
args = get_args()
if args.arg_log:
    write_config_file(args)

print("Setting random seed...")
pl.seed_everything(args.seed)

embeddings_file = Path(args.embeddings_file).resolve()
protection_file = Path(args.protection_file).resolve()
habitat_file = Path(args.habitat_file).resolve()

print("Calculating class weights...")
class_weights = calculate_class_weights(protection_file, args.num_classes)
print(f"Class weights: {class_weights}")

print("Initializing data module...")
datamodule = MergedDataModule(embeddings_file, protection_file, habitat_file, args.batch_size)

print("Initializing classifier model...")
sample_emb_dim = datamodule.dataset.embeddings.shape[1]
habitat_dim = datamodule.dataset.habitats.shape[1]
model = Classifier(sample_emb_dim=sample_emb_dim, habitat_dim=habitat_dim, num_classes=args.num_classes, initial_learning_rate=args.initial_learning_rate, class_weights=class_weights)

print("Setting up checkpoint directory...")
checkpoint_dir = Path('checkpoints_classifier')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print("Initializing checkpoint callback...")
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    dirpath=checkpoint_dir,
    filename='classifier-{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    mode='max',
)

print("Initializing Wandb logger...")
wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir=Path("lightning_logs"), config=args, log_model=False)

# Inizializzazione Wandb
print("Initializing Wandb run...")
wandb_run = wandb.init(project='ORDNA_Class_july', config=args)
print(f"Wandb run URL: {wandb_run.url}")

print("Initializing trainer...")
trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=10,
    detect_anomaly=False
)

print("Starting training...")
trainer.fit(model=model, datamodule=datamodule)

print("Finishing Wandb run...")
wandb.finish()
