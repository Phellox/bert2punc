import torch
from torch.utils.data import TensorDataset
from variables import PROJECT_PATH
from src.models import train_model

def load_data(path):
        X, y = torch.load(path) 
        return TensorDataset(X, y)

def test_model(i=5):
    model = train_model.TrainModel().model
    data = load_data(path=PROJECT_PATH / "data" / "processed" / "val.pt")
    with torch.no_grad():
        out = model(data.tensors[0][:i])
        y_pred = torch.argmax(out, dim=-1)
    assert y_pred.shape == data.tensors[1][:i].shape