import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from merged_dataset import MergedDataModule
from ORDNA.models.classifier import Classifier
from ORDNA.utils.argparser import get_args, write_config_file
import wandb
import pandas as pd
from pathlib import Path

# Funzione per calcolare i pesi delle classi
def calculate_class_weights_from_csv(protection_file: Path, num_classes: int) -> torch.Tensor:
    labels_df = pd.read_csv(protection_file)
    label_counts = labels_df['protection'].value_counts().sort_index()
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    return torch.tensor(class_weights.values, dtype=torch.float)

def main():
    args = get_args()
    if args.arg_log:
        write_config_file(args)

    print(f"[rank: {0}] Seed set to {args.seed}")
    pl.seed_everything(args.seed)

    datamodule = MergedDataModule(
        embeddings_file=args.embeddings_file,
        protection_file=args.protection_file,
        habitat_file=args.habitat_file,
        batch_size=args.batch_size
    )
    datamodule.setup()

    sample_emb_dim = datamodule.sample_emb_dim  # Dimensione degli embeddings
    habitat_dim = datamodule.num_habitats  # Dimensione della codifica one-hot degli habitat

    print(f"sample_emb_dim: {sample_emb_dim}, habitat_dim: {habitat_dim}")

    class_weights = calculate_class_weights_from_csv(Path(args.protection_file), args.num_classes)
    print(f"class_weights: {class_weights}")

    model = Classifier(
        sample_emb_dim=sample_emb_dim,
        num_classes=args.num_classes,
        habitat_dim=habitat_dim,
        initial_learning_rate=args.initial_learning_rate,
        class_weights=class_weights
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='checkpoints_classifier',
        filename='classifier-{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=3,
        mode='max',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max'
    )

    wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir="lightning_logs", config=args, log_model=False)
    wandb_run = wandb.init(project='ORDNA_Class_july', config=args)
    print(f"Wandb run URL: {wandb_run.url}")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10 )

    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    print(f"Early stopping triggered: {trainer.should_stop}")
    wandb.finish()

if __name__ == '__main__':
    main()
