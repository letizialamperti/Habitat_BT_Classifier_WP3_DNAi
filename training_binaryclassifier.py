import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ORDNA.data.barlow_twins_datamodule import BarlowTwinsDataModule
from ORDNA.models.binary_classifier import BinaryClassifier
from ORDNA.utils.argparser import get_args, write_config_file
import wandb

# Funzione per calcolare i pesi delle classi
def calculate_class_weights_from_csv(labels_file: Path) -> torch.Tensor:
    import pandas as pd
    labels = pd.read_csv(labels_file)
    label_counts = labels['protection'].value_counts().sort_index()
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum() * 2  # Normalize weights for binary classification
    return torch.tensor(class_weights.values, dtype=torch.float)

def main():
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

    samples_dir = Path(args.samples_dir).resolve()

    print("Initializing data module...")
    datamodule = BarlowTwinsDataModule(
        samples_dir=samples_dir,
        labels_file=Path(args.labels_file).resolve(),
        sequence_length=args.sequence_length,
        sample_subset_size=args.sample_subset_size,
        batch_size=args.batch_size
    )

    print("Setting up data module...")
    datamodule.setup(stage='fit')  # Ensure train_dataset is defined

    print("Calculating class weights from CSV...")
    class_weights = calculate_class_weights_from_csv(Path(args.labels_file).resolve())
    print(f"Class weights: {class_weights}")

    print("Initializing binary classifier model...")
    # Crea il classificatore binario
    sample_emb_dim = args.sample_emb_dim  # Assicurati che questo sia corretto
    model = BinaryClassifier(
        sample_emb_dim=sample_emb_dim,
        initial_learning_rate=args.initial_learning_rate,
        class_weights=class_weights
    )

    print("Setting up checkpoint directory...")
    # Checkpoint directory
    checkpoint_dir = Path('checkpoints_binary_classifier')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing checkpoint callback...")
    # General checkpoint callback for best model saving
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',  # Ensure this metric is logged in your model
        dirpath=checkpoint_dir,
        filename='binary_classifier-{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=3,
        mode='max',
    )

    print("Setting up Wandb logger...")
    # Setup logger e trainer
    wandb_logger = WandbLogger(project='ORDNA_Class_july_binary', save_dir=Path("lightning_logs"), config=args, log_model=False)

    # Inizializzazione Wandb
    print("Initializing Wandb run...")
    wandb_run = wandb.init(project='ORDNA_Class_july_binary', config=args)

    # Print Wandb run URL
    print(f"Wandb run URL: {wandb_run.url}")

    print("Initializing trainer...")

    # Parametri del dataset e batch size
    N = len(datamodule.train_dataloader().dataset)  # Numero di campioni di addestramento
    B = args.batch_size  # Batch size

    # Calcolare il numero totale di batch per epoca
    num_batches_per_epoch = N // B
    print(f"Number of batches per epoch: {num_batches_per_epoch}")

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor='val_accuracy', patience=5, mode='max')],
        log_every_n_steps=10)

    # Ensure the callbacks are correctly passed
    print(f"Trainer callbacks: {trainer.callbacks}")

    # Start training
    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    # Check if early stopping was triggered
    if trainer.should_stop:
        stopped_epoch = trainer.current_epoch
        print(f"Early stopping was triggered at epoch {stopped_epoch}.")

    # Chiudi Wandb
    print("Finishing Wandb run...")
    wandb.finish()

if __name__ == '__main__':
    main()
