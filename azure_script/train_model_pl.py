import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch.cuda
from torch.utils.data import TensorDataset, DataLoader
import torch
from pathlib import Path

#from src.data.load_data import load_dataset, create_dataloader
from src.models.model_pl import BERT_Model
#from variables import PROJECT_PATH

def load_dataset(set_type: str = "train", dir_path= Path('.').parent / 'data' / "processed"):
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

if __name__ == '__main__':
    # Define datasets and data loaders
    train_set = load_dataset("train")
    val_set = load_dataset("val")

    batch_size = 32
    num_workers = 4

    train_loader = create_dataloader(train_set, batch_size, True, num_workers)
    val_loader = create_dataloader(val_set, batch_size, False, num_workers)

    # Define model
    output_size = 4
    segment_size = 32
    dropout = 0.3
    model = BERT_Model(segment_size, output_size, dropout)

    # Define trainer
    save_dir = Path('.') / "models"
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(max_epochs=5, limit_train_batches=10, limit_val_batches= 10, default_root_dir=save_dir, gpus = 1, logger=tb_logger)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    # gpus = 1
