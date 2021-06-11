import transformers
import torch
from torch import nn
from absl import flags


class BERT_Model(nn.Module):
    def __init__(self, modeltype = 'bert-base-cased'):
        super().__init__()
        self.model = transformers.BertModel.from_pretrained(modeltype)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
