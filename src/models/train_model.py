'''
Finetune the pre-trained model
Bert-Base-uncased: 12-layer, 768-hidden, 12-heads, 109M parameters.
Trained on cased English text.
'''

import sys
import argparse
from tqdm.auto import tqdm

import torch
from torch.cuda.random import manual_seed
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import transformers
import logging
from datasets import load_metric

from variables import PROJECT_PATH
from src.models.model import BERT_Model

logging.basicConfig(level=logging.INFO)
logger = logging

class TrainModel(object):
    def __init__(self):
        # Hyper parameters
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--epochs', default=1)
        parser.add_argument('--batch_size', default=32)
        parser.add_argument('--lr', default=1e-3)
        parser.add_argument('--momentum', default=0.9)
        parser.add_argument('--seq_length', default=32)
        parser.add_argument('--all_data', default=False, help='Set true if all data should be loaded')
        parser.add_argument('--subset_data', default=1800, help='For loading data')
        parser.add_argument('--val_split', default=0.2)
        parser.add_argument('--test_split', default=0.2)

        self.args = parser.parse_args(sys.argv[2:])

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
        self.optimizer = transformers.AdamW(params=model.parameters(), lr=float(self.args.lr))

    def load_data(self, path):
        X, y = torch.load(path)
        data = TensorDataset(X, y)
        batch_size = int(self.args.batch_size)
        # validation_split = FLAGS.val_split
        shuffle_dataset = True
        return DataLoader(data, batch_size, shuffle_dataset)

    def train_custom(self, train = True):

        train_dataloader = self.load_data(path=PROJECT_PATH / "data" / "processed" / "train.pt")
        val_dataloader = self.load_data(path=PROJECT_PATH / "data" / "processed" / "val.pt")
        test_dataloader = self.load_data(path=PROJECT_PATH / "data" / "processed" / "test.pt")
        model = self.model
        optimizer = self.optimizer

        num_epochs = int(self.args.epochs)
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
        for _ in range(num_epochs):
            for X, y in val_dataloader:
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


if __name__ == '__main__':
    train_model = TrainModel()
    train_model.train_custom()


'''
from transformers import pipeline, AutoTokenizer, BartForConditionalGeneration
#load tokenizer to preprocess our data
tokenizer = AutoTokenizer.from_pretrained(model_type)
#load the pre-trained model
model_type = 'bert-base-cased'
model = BartForConditionalGeneration.from_pretrained(model_type, max_length=50)
#define tokenizer
restore = pipeline(model_type, model=model, tokenizer=tokenizer)

#Define Trainer (a simple but feature-complete training and eval loop for PyTorch)

punctuations = "?!,-:;."

from transformers import TrainingArguments

training_args = TrainingArguments("test_trainer")

from transformers import Trainer

trainer = Trainer(
    model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
)

#fine-tune model; start training
trainer.train()
'''
