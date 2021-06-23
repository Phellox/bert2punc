import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from transformers import BertForMaskedLM


class BERT_Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.lr = hparams.lr
        self.output_size = hparams.output_size
        confusion_matrix = pl.metrics.ConfusionMatrix(hparams.output_size)
        self.conf_mat_val = confusion_matrix.clone()
        self.conf_mat_test = confusion_matrix.clone()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.bert_vocab_size = 30522
        self.bn = nn.BatchNorm1d(hparams.segment_size * self.bert_vocab_size)
        self.fc = nn.Linear(hparams.segment_size * self.bert_vocab_size, hparams.output_size)
        self.dropout = nn.Dropout(hparams.dropout)

        for param in self.bert.parameters():
            param.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BERTModel")
        parser.add_argument('--segment_size', type=int, default=32)
        parser.add_argument('--output_size', type=int, default=3)
        parser.add_argument('--dropout', type=float, default=0.398)
        parser.add_argument('--lr', type=float, default=0.0029)
        return parent_parser

    def forward(self, x):
        if type(x) == list:
            x, _ = x
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

        return loss, pred, labels

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.eval_step(batch, batch_idx)
        self.conf_mat_val(pred, labels)
        return loss

    def validation_epoch_end(self, loss):
        metrics = {"val_loss": torch.tensor(loss).mean()}

        conf_mat = self.conf_mat_val.compute()
        metrics["val_acc"] = conf_mat.trace() / conf_mat.sum()
        for i in range(self.output_size):
            metric_name = "val_acc_class_{}".format(i)
            metrics[metric_name] = conf_mat[i, i] / conf_mat[i, :].sum()

        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        loss, pred, labels = self.eval_step(batch, batch_idx)
        self.conf_mat_test(pred, labels)
        return loss

    def test_epoch_end(self, loss):
        metrics = {"test_loss": torch.tensor(loss).mean()}

        conf_mat = self.conf_mat_test.compute()
        metrics["test_acc"] = conf_mat.trace() / conf_mat.sum()
        for i in range(self.output_size):
            metric_name = "test_acc_class_{}".format(i)
            metrics[metric_name] = conf_mat[i, i] / conf_mat[i, :].sum()

        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }