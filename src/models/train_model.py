'''
Finetune the pre-trained model
Bert-Base-uncased: 12-layer, 768-hidden, 12-heads, 109M parameters.
Trained on cased English text.
'''

from torch.cuda.random import manual_seed
import numpy as np
from absl import flags, app
from tqdm.auto import tqdm

import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
import logging
from datasets import load_metric, load_from_disk
logging.basicConfig(level=logging.INFO)
logger = logging

FLAGS = flags.FLAGS

# Hyper parameters
#cased makes a difference between case and lowercase
flags.DEFINE_string('model_type', 'bert-base-cased', 'Pretrained model')
flags.DEFINE_integer('epochs', 2, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_boolean('all_data',False,'Set true if all data should be loaded')
flags.DEFINE_integer('subset_data',1800,'For loading data')
flags.DEFINE_float('val_split',0.2,'')
flags.DEFINE_float('test_split', 0.2,'')

from variables import PROJECT_PATH
from src.models.model import BERT_Model

#tokenized_datasets = PROJECT_PATH / 'src' / 'data' / 'processed'


class TrainModel(object):
    def __init__(self):
        super().__init__()
        print('Model type ',FLAGS.model_type)
        model = BERT_Model(modeltype=FLAGS.model_type)
        self.path = '../trained_model.pt'
        try:
            checkpoint = torch.load(self.path)
            model = model.load_state_dict(checkpoint)
            #print(model)
            #print(checkpoint)
        except OSError as e:
            model = model
            print('No preloaded model')
            #print(model)

        self.criterion = nn.NLLLoss()
        # Use GPU if we can to make training faster
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Initialize optimizer")
        self.optimizer = transformers.AdamW(params=model.parameters(), lr=FLAGS.lr)

        self.model = model
        self.tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model_type)

    def load_data(self, custom = False):
        path = PROJECT_PATH / 'src' / 'data' / 'processed'
        tokenized_datasets = load_from_disk(str(path))
        batch_size = FLAGS.batch_size
        validation_split = FLAGS.val_split
        shuffle_dataset = True
        random_seed = 42
        dataset_size = len(tokenized_datasets["train"])


        if not custom:

            train_dataset, eval_dataset = random_split(
                tokenized_datasets["train"],
                [int(dataset_size * (validation_split)), dataset_size - int(dataset_size * (validation_split))],
                generator=torch.Generator().manual_seed(42)
            )

            return train_dataset, eval_dataset

        else:
            indices = list(range(dataset_size))
            dataset = tokenized_datasets["train"].shuffle(seed=random_seed).select(indices)

            # Creating data indices for training and validation splits:

            split = int(np.floor(validation_split * dataset_size))
            np.random.seed(random_seed)
            np.random.shuffle(indices)


            train_indices, val_indices = indices[split:], indices[:split]
            return train_indices, val_indices
            '''
            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                       sampler=train_sampler)
            eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                        sampler=valid_sampler)
        #features: ['attention_mask', 'input_ids', 'text', 'title', 'token_type_ids']
        #print(next(iter(train_dataloader))['title'])
            return train_dataloader, eval_dataloader
            '''


    def train_simpel(self):
        print('Loading dataset')
        train_dataloader, eval_dataloader = self.load_data(custom= False)

        training_args = transformers.TrainingArguments("test_trainer")
        '''
        training_args = transformers.TrainingArguments(
            output_dir="output",
            evaluation_strategy="steps",
            eval_steps=500,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            save_steps=3000,
            seed=0,
            load_best_model_at_end=True,
        )
        '''
        '''
         do_train=True,
            num_train_epochs = FLAGS.epochs,
            learning_rate= FLAGS.lr,
            logging_dir=logger,
            output_dir= 'output',
            load_best_model_at_end=True
            '''


        #TypeError: _forward_unimplemented() got an unexpected keyword argument 'attention_mask'
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataloader,
            eval_dataset=eval_dataloader
        )
        #finetune model
        trainer.train()

        accuracy = load_metric('accuracy')

    def train_custom(self):
        train_dataloader, eval_dataloader = self.load_data(custom = True)
        model = self.model
        optimizer = self.optimizer

        num_epochs = FLAGS.epochs
        num_training_steps = num_epochs * len(train_dataloader)

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
            for batch in train_dataloader:
                batch = {k: v for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        #evaluate
        metric = load_metric("accuracy")
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        final_score = metric.compute()
        print('Final score: ', final_score)

        '''
        #If model is better, save the trained model
        if  final_score['accuracy'] > best_accuracy:
            best_accuracy = final_score['accuracy']
            torch.save(model.state_dict(), self.path)
        '''


def main(argv):
    # TrainOREvaluate().train()
    # writer.close()
    #TrainModel().train_custom()
    TrainModel().train_simpel()
    #TrainModel().load_data()

if __name__ == '__main__':
    app.run(main)


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