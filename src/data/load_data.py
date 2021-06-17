from torch.utils.data import TensorDataset, DataLoader
import torch

from variables import PROJECT_PATH


def load_dataset(set_type: str = "train", dir_path=PROJECT_PATH / "data" / "processed"):
    set_type = set_type if set_type.endswith(".pt") else set_type + ".pt"
    path = dir_path / set_type
    if path.exists():
        X, y = torch.load(path)
        data = TensorDataset(X, y)
    else:
        raise ValueError("The path: {}, did not lead to a proper .pt file.".format(path))

    return data


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
