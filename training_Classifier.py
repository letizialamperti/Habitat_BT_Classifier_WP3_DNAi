import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from merged_dataset import MergedDataModule
from ORDNA.models.classifier import Classifier
from ORDNA.utils.argparser import get_args, write_config_file
import wandb

# Funzione per calcolare i pesi delle classi
def calculate_class_weights(labels_file: Path, num_classes: int) -> torch.Tensor:
    import pandas as pd
    labels = pd.read_csv(labels_file)
    label_counts = labels['protection'].value_counts().sort_index()
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    return torch.tensor(class_weights.values, dtype=torch.float)

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
            val_dataloader = trainer.datamodule.val_dataloader()
            val_class_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    embeddings, habitats, labels = batch
                    embeddings, habitats, labels = embeddings.to(pl_module.device), habitats.to(pl_module.device), labels.to(pl_module.device)
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

def main():
    # Parsing arguments
    args = get_args()
    if args.arg_log:
        write_config_file(args)

    print("Setting random seed...")
    pl.seed_everything(args.seed)

    print("Setting up data module...")
    datamodule = MergedDataModule(
        embeddings_file=args.embeddings_file,
        protection_file=args.protection_file,
        habitat_file=args.habitat_file,
        batch_size=args.batch_size
    )

    print("Calculating class weights from CSV...")
    class_weights = calculate_class_weights(Path(args.protection_file).resolve(), args.num_classes)
    print(f"Class weights: {class_weights}")

    print("Initializing classifier model...")
    sample_emb_dim = datamodule.dataset.embeddings.shape[1]  # Dimensione degli embeddings
    habitat_dim = datamodule.dataset.habitats.shape[1]  # Dimensione dell'habitats
    model = Classifier(sample_emb_dim=sample_emb_dim, num_classes=args.num_classes, num_habitats=habitat_dim, initial_learning_rate=args.initial_learning_rate, class_weights=class_weights)

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

    print("Setting up Wandb logger...")
    wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir=Path("lightning_logs"), config=args, log_model=False)

    print("Initializing Wandb run...")
    wandb_run = wandb.init(project='ORDNA_Class_july', config=args)

    print(f"Wandb run URL: {wandb_run.url}")

    print("Initializing trainer...")
    num_batches_per_epoch = len(datamodule.train_dataloader())
    print(f"Number of batches per epoch: {num_batches_per_epoch}")

    n_steps = num_batches_per_epoch // 10
    print(f"Validation will run every {n_steps} steps")

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

    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    if any(callback.stop_training for callback in trainer.callbacks if isinstance(callback, CustomEarlyStopping)):
        stopped_epoch = next(callback.stopped_epoch for callback in trainer.callbacks if isinstance(callback, CustomEarlyStopping))
        print(f"Early stopping was triggered at epoch {stopped_epoch}.")

    print("Finishing Wandb run...")
    wandb.finish()

if __name__ == "__main__":
    main()
