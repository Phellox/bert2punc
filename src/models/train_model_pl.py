import pytorch_lightning as pl
from argparse import ArgumentParser

from src.data.load_data import load_dataset, create_dataloader
from src.models.model_pl import BERT_Model
from variables import PROJECT_PATH

if __name__ == '__main__':
    # Define parser
    parser = ArgumentParser()

    # Add PROGRAM level args (None currently)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)

    # Add MODEL specific args
    parser = BERT_Model.add_model_specific_args(parser)

    # Add TRAINER specific args
    parser = pl.Trainer.add_argparse_args(parser)
    save_dir = PROJECT_PATH / "models"
    parser.set_defaults(default_root_dir=str(save_dir))

    # Parse args
    hparams = parser.parse_args()

    # Define trainer
    trainer = pl.Trainer.from_argparse_args(hparams)

    # Define model
    model = BERT_Model(hparams)

    # Define datasets and data loaders
    train_set = load_dataset("train")
    val_set = load_dataset("val")
    train_loader = create_dataloader(train_set, hparams.batch_size, hparams.shuffle, hparams.num_workers)
    val_loader = create_dataloader(val_set, hparams.batch_size, False, hparams.num_workers)

    # Train model   # TODO: Add early stopping
    trainer.fit(model, train_loader, val_loader)
