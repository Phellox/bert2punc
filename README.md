BERT2Punc
==============================

This project aims to use the BERT model to predict where to insert punctuations in sequences of text. Initially, the model will be based on the pretrained BERT model from the Transformers repository created by Hugging Face: https://github.com/huggingface/transformers. Similarly, the initial dataset will be english texts from Wikipedia loaded through the Transformers repository. If time allows for it, we will also try to use a pretrained Nordic BERT model from https://github.com/botxo/nordic_bert to predict punctuation in danish texts from Wikipedia.

# Running the code

To create an optimal environment to run the code, just go to the requirements.txt file to see what dependencies it requires. 

## Load the data
There next you will need to load the data. For this just go to make_dataset.py and run the script. It will then download the raw data with English Wikipedia articles from datasets into the ‘raw’ folder. Then the data will be processed by splitting the data into a training, validation and testing dataset. Those will all be processed to give an input of text with a word followed by a whitespace and an output of the text and the true character (punctuation, comma or whitespace). Then the files will be saved as a PyTorch file (.pt) and save in the ‘processed’ folder. 

## Train the model
The model comes in two different version: As a normal PyTorch base class (model.py) and as a PyTorch LightningModule (model_pl.py). Both models are based on the pre-trained Bert model.
Both models have each their training script, train_model.py and train_model_pl.py that can be run as they are. Since the training data is quite big, it can be recommended considering starting to run with only the validation da ta for the training and then split that one into a training- and validation dataset. The hyperparameters can be changed in the command-line. Else they can be optimized by in the PyTorch Lightning to add the argument ``--optimise, True`` to start running Optuna https://optuna.readthedocs.io/en/stable/index.html. 
While running the train_model_pl.py file the train loss for each batch, validation loss for each epoch, validation accuracy for each epoch and the per class accuracy per epoch will be logged together with the used hyperparameters. 



Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── azure_script
    │   ├── Run Bert2Punc in Azure.ipynb    <- Script with guide to train the model in Azure
    │   ├── azure_train_model_pl.py     <- Training script changed to fit Azure
    │   └── conda_dependencies.yml      <- Dependencies needed to create environment in Azure
    │
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
