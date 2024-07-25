import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall

class Classifier(pl.LightningModule):
    def __init__(self, sample_emb_dim: int, num_classes: int, num_habitats: int, initial_learning_rate: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(sample_emb_dim + num_habitats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        ).to(self.device)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        embeddings, habitats = batch
        combined_input = torch.cat((embeddings, habitats), dim=1)

        output = self(combined_input)
        labels = batch[2].to(self.device)  # Assuming labels are the third element in the batch
        class_loss = self.loss_fn(output, labels)
        
        self.log('train_class_loss', class_loss)
        pred = torch.argmax(output, dim=1)
        accuracy = self.train_accuracy(pred, labels)
        self.log('train_accuracy', accuracy)
        precision = self.train_precision(pred, labels)
        self.log('train_precision', precision)
        recall = self.train_recall(pred, labels)
        self.log('train_recall', recall)
        return class_loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        embeddings, habitats = batch
        combined_input = torch.cat((embeddings, habitats), dim=1)

        output = self(combined_input)
        labels = batch[2].to(self.device)  # Assuming labels are the third element in the batch
        class_loss = self.loss_fn(output, labels)
        
        self.log('val_class_loss', class_loss)
        pred = torch.argmax(output, dim=1)
        accuracy = self.val_accuracy(pred, labels)
        self.log('val_accuracy', accuracy)
        precision = self.val_precision(pred, labels)
        self.log('val_precision', precision)
        recall = self.val_recall(pred, labels)
        self.log('val_recall', recall)
        return class_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True),
            'monitor': 'val_class_loss'
        }
        return [optimizer], [scheduler]
