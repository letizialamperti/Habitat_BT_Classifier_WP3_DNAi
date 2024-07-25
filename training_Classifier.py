import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataModule
from ORDNA.models.classifier import Classifier
from ORDNA.utils.argparser import get_args, write_config_file
import wandb

# Funzione per calcolare i pesi delle classi
def calculate_class_weights_from_csv(labels_file: Path, num_classes: int) -> torch.Tensor:
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
labels_file = Path(args.labels_file).resolve()
habitats_file = Path(args.habitats_file).resolve()

print("Initializing data module...")
datamodule = BarlowTwinsDataModule(
    embeddings_file=embeddings_file,
    labels_file=labels_file,
    habitats_file=habitats_file,
    batch_size=args.batch_size
)

print("Setting up data module...")
datamodule.setup(stage='fit')

print("Calculating class weights from CSV...")
class_weights = calculate_class_weights_from_csv(labels_file, args.num_classes)
print(f"Class weights: {class_weights}")

print("Initializing classifier model...")
model = Classifier(
    sample_emb_dim=args.sample_emb_dim, 
    habitat_emb_dim=args.habitat_emb_dim, 
    num_classes=args.num_classes, 
    initial_learning_rate=args.initial_learning_rate,
    class_weights=class_weights
)

print("Setting up checkpoint directory...")
checkpoint_dir = Path('checkpoints_classifier')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print("Initializing checkpoint callback...")
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',  # Ensure this metric is logged in your model
    dirpath=checkpoint_dir,
    filename='classifier-{epoch:02d}-{val_accuracy:.2f}',
    save_top_k=3,
    mode='max',
)

print("Setting up Wandb logger...")
wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir=Path("lightning_logs"), config=args, log_model=False)

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

print(f"Trainer callbacks: {trainer.callbacks}")

print("Starting training...")
trainer.fit(model=model, datamodule=datamodule)

print("Finishing Wandb run...")
wandb.finish()
