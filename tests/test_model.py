import torch
from variables import PROJECT_PATH
from src.models import train_model
from src.data.load_data import load_dataset

def test_model_output(i=5):
    model = train_model.TrainModel().model
    data = load_dataset(set_type = 'val', dir_path=PROJECT_PATH / "data" / "processed")
    with torch.no_grad():
        out = model(data.tensors[0][:i])
        y_pred = torch.argmax(out, dim=-1)
    assert type(y_pred) == type(data.tensors[1])
    assert y_pred.shape == data.tensors[1][:i].shape