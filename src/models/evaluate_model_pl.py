import pytorch_lightning as pl
from argparse import ArgumentParser

from src.data.load_data import load_dataset, create_dataloader
from src.models.model_pl import BERT_Model
from variables import PROJECT_PATH

if __name__ == '__main__':
    # Define parser
    parser = ArgumentParser()

    # Add PROGRAM level args (None currently)
    parser.add_argument('--model_path', type=str)
    # parser.add_argument('--data_path', type=str, default=str(PROJECT_PATH / "data" / "processed" / "test.pt"))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Add TRAINER specific args
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse args
    args = parser.parse_args()

    # Define trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # Define model
    model = BERT_Model.load_from_checkpoint(args.model_path)

    # Define datasets and data loaders
    test_set = load_dataset("test")
    test_loader = create_dataloader(test_set, args.batch_size, False, args.num_workers)

    # Evaluate model performance on test set
    trainer.test(model, test_loader)
