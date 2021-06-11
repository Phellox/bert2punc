''''
Finetune the pre-trained model
Bert-Base-uncased: 12-layer, 768-hidden, 12-heads, 109M parameters.
Trained on cased English text.
'''

from absl import flags, app
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
import logging
from datasets import load_metric
logging.basicConfig(level=logging.INFO)
logger = logging

FLAGS = flags.FLAGS

# Hyper parameters
flags.DEFINE_string('model_type', 'bert-base-cased', 'Pretrained model')
flags.DEFINE_integer('epochs', 2, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
#cased makes a difference between case and lowercase
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('subset_data',1000,'For loading data')

from variables import PROJECT_PATH
from src.models.model import BERT_Model

#tokenized_datasets = PROJECT_PATH / 'data' / 'processed'

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
            print(model)

        self.criterion = nn.NLLLoss()
        #Use GPU if we can to make training faster
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('Initialize optimizer')
        self.optimizer = transformers.AdamW(params=model.parameters(),
                                            lr = FLAGS.lr)

        self.model = model


    def load_data(self, custom = False):
        tokenized_datasets = PROJECT_PATH / 'data' / 'processed'
        if custom:
            tokenized_datasets = tokenized_datasets.remove_columns(['text'])
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            tokenized_datasets.set_format("torch")

        train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(FLAGS.subset_data))
        eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(FLAGS.subset_data))

        if custom:
            train_dataset = DataLoader(train_dataset, shuffle=True, batch_size=FLAGS.batch_size)
            eval_dataset = DataLoader(eval_dataset, batch_size=FLAGS.batch_size)

        return train_dataset, eval_dataset


    def train_simpel(self):
        train_dataset, eval_dataset = self.load_data(custom = False)

        training_args = transformers.TrainingArguments(
            do_train=True,
            num_train_epochs = FLAGS.epochs,
            learning_rate= FLAGS.lr,
            no_cuda=not torch.cuda.is_available(),
            load_best_model_at_end=True,
            logging_dir=logger
        )

        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        #finetune model
        trainer.train()

        accuracy = load_metric('accuracy')

    def train_custom(self):
        train_dataloader, eval_dataloader = self.load_data(custom=True)
        model = self.model
        optimizer = self.optimizer

        num_epochs = FLAGS.epochs
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
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
                batch = {k: v.to(device) for k, v in batch.items()}
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
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        final_score = metric.compute()

        '''
        #If model is better, save the trained model
        if  final_score['accuracy'] > best_accuracy:
            best_accuracy = final_score['accuracy']
            torch.save(model.state_dict(), self.path)
        '''


def main(argv):
    # TrainOREvaluate().train()
    # writer.close()
    TrainModel().train_simpel()

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