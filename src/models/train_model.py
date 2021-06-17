'''
Finetune the pre-trained model
Bert-Base-uncased: 12-layer, 768-hidden, 12-heads, 109M parameters.
Trained on cased English text.
'''

from absl import flags, app
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import transformers
import logging
from datasets import load_metric

from variables import PROJECT_PATH
from src.models.model import BERT_Model

logging.basicConfig(level=logging.INFO)
logger = logging

FLAGS = flags.FLAGS

# Hyper parameters
flags.DEFINE_integer('epochs', 1, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('lr', 1e-3, '')


class TrainModel(object):
    def __init__(self):
        super().__init__()
        output_size = 4
        segment_size = 32
        dropout = 0.3

        model = BERT_Model(segment_size, output_size, dropout)
        self.path = '../trained_model.pt'
        try:
            checkpoint = torch.load(self.path)
            model = model.load_state_dict(checkpoint)
        except OSError as e:
            model = model
            print('No preloaded model')

        self.criterion = nn.CrossEntropyLoss()
        # Use GPU if we can to make training faster
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model

        print("Initialize optimizer")
        self.optimizer = transformers.AdamW(params=model.parameters(), lr=FLAGS.lr)

    def load_data(self, path):
        X, y = torch.load(path)
        data = TensorDataset(X, y)
        batch_size = FLAGS.batch_size
        shuffle_dataset = True
        return DataLoader(data, batch_size, shuffle_dataset)

    def train_custom(self, train = True):

        train_dataloader = self.load_data(path=PROJECT_PATH / "data" / "processed" / "train.pt")
        val_dataloader = self.load_data(path=PROJECT_PATH / "data" / "processed" / "val.pt")
        test_dataloader = self.load_data(path=PROJECT_PATH / "data" / "processed" / "test.pt")
        model = self.model
        optimizer = self.optimizer

        if train:
            num_epochs = FLAGS.epochs
            num_training_steps = num_epochs * len(val_dataloader)

            lr_scheduler = transformers.get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
            )

            #add a progress bar over our number of training steps
            progress_bar = tqdm(range(num_training_steps))

            model.train()
            for epoch in range(num_epochs):
                for X, y in val_dataloader:
                    X = X
                    y = y
                    outputs = model(X)
                    loss = self.criterion(outputs, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    progress_bar.update(1)

            #evaluate
        metric = load_metric("accuracy")
        model.eval()
        for X, y in test_dataloader:
            with torch.no_grad():
                outputs = model(X)

            predictions = torch.argmax(outputs, dim=-1)
            metric.add_batch(predictions=predictions, references=y)
        final_score = metric.compute()
        print('Final score: ', final_score)

        '''
        #If model is better, save the trained model
        if  final_score['accuracy'] > best_accuracy:
            best_accuracy = final_score['accuracy']
            torch.save(model.state_dict(), self.path)
        '''


def main(argv):
    TrainModel().train_custom()


if __name__ == '__main__':
    app.run(main)
