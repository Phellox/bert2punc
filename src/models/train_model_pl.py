import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import yaml

from src.data.load_data import load_dataset, create_dataloader
from src.models.model_pl import BERT_Model
from variables import PROJECT_PATH

if __name__ == '__main__':
    # Define parser
    parser = ArgumentParser()

    # Add PROGRAM level args (None currently)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--optimise', action='store_true')
    #parser.add_argument('--seed', default=42)

    pl.utilities.seed.seed_everything(seed=42, workers=False) #TODO: Include parser

    # Add MODEL specific args
    parser = BERT_Model.add_model_specific_args(parser)

    # Add TRAINER specific args
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse args
    hparams = parser.parse_args()

    if hparams.optimise:
        def objective(trial) -> float:

            #hparams.shuffle = trial.suggest_int('shuffle', 0, 1)
            #hparams.batch_size = trial.suggest_int('batch_size', 16, 128)
            hparams.dropout = trial.suggest_uniform('dropout', 0, 0.75)
            hparams.lr = trial.suggest_loguniform('learning_rate', 1e-6, 1)

            # Define trainer
            trainer = pl.Trainer.from_argparse_args(hparams,
             checkpoint_callback=False,
             callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
             weights_summary=None)

            hyperparameters = dict(lr=hparams.lr, dropout=hparams.dropout)
            trainer.logger.log_hyperparams(hyperparameters)

            # Define model
            model = BERT_Model(hparams)

            # Define datasets and data loaders
            train_set = load_dataset("train")
            val_set = load_dataset("val")
            train_loader = create_dataloader(train_set, hparams.batch_size, bool(hparams.shuffle), hparams.num_workers)
            val_loader = create_dataloader(val_set, hparams.batch_size, False, hparams.num_workers)

            # save_dir = PROJECT_PATH / "models"
            # Train model   # TODO: Add early stopping
            trainer.fit(model, train_loader, val_loader)

            return trainer.callback_metrics["val_acc"].item()
            
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=20, interval_steps=10)
            )
        study.optimize(objective, n_trials=10)

        fig = plt.figure()
        fig.add_axes(optuna.visualization.matplotlib.plot_parallel_coordinate(study), label='Parallel coordinate')
        plt.savefig('Parallel coordinate.png')
        fig = plt.figure()
        fig.add_axes(optuna.visualization.matplotlib.plot_optimization_history(study), label='Optimisation history')
        plt.savefig('Optimisation history.png')

        print('Best parameters: ', study.best_params)
        with open('best_params.yml', 'w') as outfile:
            yaml.dump(study.best_params, outfile, default_flow_style=False)

    else:
        # Define trainer
        trainer = pl.Trainer.from_argparse_args(hparams)

        # Define model
        model = BERT_Model(hparams)

        # Define datasets and data loaders
        train_set = load_dataset("train")
        val_set = load_dataset("val")
        train_loader = create_dataloader(train_set, hparams.batch_size, hparams.shuffle, hparams.num_workers)
        val_loader = create_dataloader(val_set, hparams.batch_size, False, hparams.num_workers)

        # save_dir = PROJECT_PATH / "models"
        # Train model   # TODO: Add early stopping
        trainer.fit(model, train_loader, val_loader)
