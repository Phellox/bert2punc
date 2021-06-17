import pytorch_lightning as pl
import torch.cuda

from src.data.load_data import load_dataset, create_dataloader
from src.models.model_pl import BERT_Model
from variables import PROJECT_PATH

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
    save_dir = PROJECT_PATH / "models"
    trainer = pl.Trainer(max_epochs=100, limit_train_batches=0.2, default_root_dir=save_dir)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
