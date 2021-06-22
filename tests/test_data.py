from src.data.make_dataset import insert_target
import torch
import pytest
from variables import PROJECT_PATH

@pytest.mark.parametrize("data_set", ["train.pt", "val.pt", "test.pt"])
def test_datasets(data_set):
    X, y = torch.load(str(PROJECT_PATH / "data" / "processed" / data_set))
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[1] == 32
    assert X.shape[0] == y.shape[0]
