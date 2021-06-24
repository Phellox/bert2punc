import logging

import torch
import transformers
from absl import flags
from datasets import load_metric
from torch import nn

# Load data, model, and project path
from src.models.model import BERT_Model
from variables import PROJECT_PATH


"""
Load pretrained model
Use the model to get a prediction about where there should be a punctuation in a given text
"""

"""'
Finetune the pre-trained model
Bert-Base-uncased: 12-layer, 768-hidden, 12-heads, 109M parameters.
Trained on cased English text.
"""

logging.basicConfig(level=logging.INFO)
FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 3, "")
flags.DEFINE_integer("batch_size", 8, "")
flags.DEFINE_float("lr", 1e-2, "")
flags.DEFINE_float("momentum", 0.9, "")
# cased makes a difference between case and lowercase
flags.DEFINE_string("model", "bert-base-cased", "Pretrained model")
flags.DEFINE_integer("seq_length", 32, "")
flags.DEFINE_integer("subset_data", 1000, "For loading data")
flags.DEFINE_string("metrics", "accuracy", "")

tokenized_datasets = PROJECT_PATH / "data" / "processed"


class EvalModel(object):
    def __init__(self):
        super().__init__()
        model = BERT_Model
        self.path = "../trained_model.pt"
        try:
            checkpoint = torch.load(self.path)
            self.model = model.load_state_dict(checkpoint)
            # print(model)
            # print(checkpoint)
        except OSError as e:
            self.model = model

        self.criterion = nn.NLLLoss()
        # Use GPU if we can to make training faster
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.optimizer = transformers.AdamW(self.model.parameters(), lr=FLAGS.lr)

    def load_data(self, custom=False):

        if custom:
            tokenized_datasets = tokenized_datasets.remove_columns(["text"])
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            tokenized_datasets.set_format("torch")

        train_dataset = (
            tokenized_datasets["train"]
            .shuffle(seed=42)
            .select(range(FLAGS.subset_data))
        )
        eval_dataset = (
            tokenized_datasets["test"].shuffle(seed=42).select(range(FLAGS.subset_data))
        )

        return train_dataset, eval_dataset

    def evaluate_model(self):
        train_dataset, eval_dataset = self.load_data()

        training_args = transformers.TrainingArguments(
            do_train=False, do_predict=True, evaluation_strategy="epoch"
        )

        metric = load_metric(FLAGS.metrics)

        def _compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=_compute_metrics,
        )
        # finetune model
        trainer.evaluate()
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.x
        np.allclose(preds, x)

    def evaluate_model_custom(self):
        train_dataset, eval_dataset = self.load_data(custom=True)
        metric = load_metric(FLAGS.metrics)

        for batch in eval_dataset:
            model_input, labels = batch
            predictions = model(model_input)
            metric.add_batch(predictions, labels)

        score = metric.compute()


if __name__ == "__main__":
    # TrainOREvaluate().train()
    # writer.close()
    EvalModel().evaluate_model()
