import torch
from torch.utils.data import TensorDataset
import pytest
from variables import PROJECT_PATH

def load_data(path):
        X, y = torch.load(path) 
        return TensorDataset(X, y)

@pytest.mark.parametrize("data_set", ["train.pt", "val.pt", "test.pt"])
def test_shapes(data_set):
    data = load_data(path=PROJECT_PATH / "data" / "processed" / data_set)
    assert data.tensors[0].shape[1] == 32
    assert data.tensors[0].shape[0] == data.tensors[1].shape[0]
