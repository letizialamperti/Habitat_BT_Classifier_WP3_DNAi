import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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

# Callback for validation on each step
class ValidationOnStepCallback(pl.Callback):
    def __init__(self, n_steps):
        super().__init__()
        self.n_steps = n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_step = trainer.global_step + 1
        if current_step % self.n_steps == 0:
            print(f"[DEBUG] Running validation at step {current_step}")
            pl_module.eval()
            val_dataloader = trainer.datamodule.val_dataloader()
            val_class_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    embeddings, habitats, labels = batch
                    combined_input = torch.cat((embeddings, habitats), dim=1)
                    output = pl_module(combined_input)
                    val_class_loss += pl_module.loss_fn(output, labels).item()
                    _, pred = torch.max(output, 1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
            val_class_loss /= len(val_dataloader)
            val_accuracy = correct / total
            print(f"[DEBUG] Validation at step {current_step}: val_class_loss = {val_class_loss}, val_accuracy = {val_accuracy}")
            pl_module.train()

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

    class_weights = calculate_class_weights_from_csv(Path(args.protection_file), args.num_classes)

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

    early_stopping_callback = CustomEarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max'
    )

    wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir="lightning_logs", config=args, log_model=False)
    wandb_run = wandb.init(project='ORDNA_Class_july', config=args)
    print(f"Wandb run URL: {wandb_run.url}")

    # Calcolare il numero totale di batch per epoca
    N = len(datamodule.train_dataloader().dataset)  # Numero di campioni di addestramento
    B = args.batch_size  # Batch size
    num_batches_per_epoch = N // B
    n_steps = max(1, num_batches_per_epoch // 30)
    print(f"Validation will run every {n_steps} steps")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, ValidationOnStepCallback(n_steps=n_steps)],
        log_every_n_steps=10,
    )

    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)
    wandb.finish()

if __name__ == '__main__':
    main()
