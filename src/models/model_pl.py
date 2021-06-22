import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
from transformers import BertForMaskedLM


class BERT_Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.lr = hparams.lr
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.bert_vocab_size = 30522
        self.bn = nn.BatchNorm1d(hparams.segment_size * self.bert_vocab_size)
        self.fc = nn.Linear(hparams.segment_size * self.bert_vocab_size, hparams.output_size)
        self.dropout = nn.Dropout(hparams.dropout)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BERTModel")
        parser.add_argument('--segment_size', type=int, default=32)
        parser.add_argument('--output_size', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.3)
        return parent_parser

    def forward(self, x):
        x = self.bert(x)["logits"]
        x = x.view(x.shape[0], -1)
        x = self.fc(self.bn(x))
        return x

    def training_step(self, batch, batch_idx):
        segments, labels = batch

        # Forward pass
        out = self.bert(segments)["logits"]
        out = out.view(out.shape[0], -1)
        out = self.fc(self.dropout(self.bn(out)))

        # Determine loss
        loss = F.cross_entropy(out, labels)

        # Log loss to TensorBoard
        self.log('train_loss', loss, on_step=True)
        return loss

    def eval_step(self, batch, batch_idx):
        segments, labels = batch

        # Forward pass
        out = self.forward(segments)

        # Determine loss
        loss = F.cross_entropy(out, labels)

        # Determine accuracy
        pred = torch.max(out, 1)[1]
        accuracy = FM.accuracy(pred, labels)

        return loss, accuracy

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.eval_step(batch, batch_idx)
        metrics = {'val_acc': accuracy, 'val_loss': loss}

        # Log loss to TensorBoard
        self.log_dict(metrics, on_epoch=True)

        return metrics

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.eval_step(batch, batch_idx)
        metrics = {'test_acc': accuracy, 'test_loss': loss}

        # Log loss to TensorBoard
        self.log_dict(metrics, on_epoch=True)

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max'),
                'monitor': 'val_acc',
                'interval': 'epoch',
                'frequency': 1
            }
        }