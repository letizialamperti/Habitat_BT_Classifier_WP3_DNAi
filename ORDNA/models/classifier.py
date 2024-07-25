import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall

class Classifier(pl.LightningModule):
    def __init__(self, sample_emb_dim: int, habitat_emb_dim: int, num_classes: int, initial_learning_rate: float = 1e-5, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(sample_emb_dim + habitat_emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        ).to(self.device)

        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)

    def forward(self, embedding: torch.Tensor, habitat: torch.Tensor) -> torch.Tensor:
        x = torch.cat((embedding, habitat), dim=1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        embeddings, habitats, labels = batch
        embeddings, habitats, labels = embeddings.to(self.device), habitats.to(self.device), labels.to(self.device)
        outputs = self(embeddings, habitats)
        loss = self.loss_fn(outputs, labels)
        self.log('train_class_loss', loss)
        preds = torch.argmax(outputs, dim=1)
        accuracy = self.train_accuracy(preds, labels)
        self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        embeddings, habitats, labels = batch
        embeddings, habitats, labels = embeddings.to(self.device), habitats.to(self.device), labels.to(self.device)
        outputs = self(embeddings, habitats)
        loss = self.loss_fn(outputs, labels)
        self.log('val_class_loss', loss)
        self.log('val_class_loss_step', loss)
        preds = torch.argmax(outputs, dim=1)
        accuracy = self.val_accuracy(preds, labels)
        self.log('val_accuracy', accuracy)
        self.log('val_accuracy_step', accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True),
            'monitor': 'val_class_loss_step'
        }
        return [optimizer], [scheduler]
