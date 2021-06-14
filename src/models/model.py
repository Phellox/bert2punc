import torch
import transformers
from absl import flags
from torch import nn


class BERT_Model(nn.Module):
    def __init__(self, modeltype="bert-base-cased"):
        super().__init__()
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(modeltype,  num_labels=2)
        print('Model loaded')
        '''
        self.model = transformers.BertModel.from_pretrained(modeltype)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)
        '''
