import torch
from torch.utils.data import TensorDataset
from absl import flags
import pytest
from variables import PROJECT_PATH
from src.models import train_model


def load_data(path):
        X, y = torch.load(path) 
        return TensorDataset(X, y)

def test_model():
    train_model.EvalModel()

print(train_model.TrainModel().model)