import torch
from torch.utils.data import TensorDataset
import pytest
from variables import PROJECT_PATH
from src.models import predict_model

def load_data(path):
        X, y = torch.load(path) 
        return TensorDataset(X, y)

def test_model():
    predict_model.EvalModel()

print(predict_model.EvalModel().model)