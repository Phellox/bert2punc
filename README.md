BERT2Punc
==============================

This project aims to use the BERT model to predict where to insert punctuations in sequences of text. Initially, the model will be based on the pretrained BERT model from the Transformers repository created by Hugging Face: https://github.com/huggingface/transformers. Similarly, the initial dataset will be english texts from Wikipedia loaded through the Datasets repository also made by HuggingFace. The implemented model and training also draws a lot of inspiration from the BertPunc github repository: https://github.com/nkrnrnk/BertPunc .

# Running the code

To create an optimal environment for running the code, install the requirements.txt file using pip. 

## Load the data
Next you will need to load the data. To do this, run make_dataset.py. It will then download the raw data, consisting of English Wikipedia articles from the Datasets repository, into the ‘raw’ data folder. Following this, the data will be processed and split into a training, validation and testing dataset. Those will all be processed to give an input of text with a word followed by a whitespace and an output of the text and the true character (punctuation, comma or whitespace). Then the files will be saved as PyTorch files (.pt) in the ‘processed’ data folder. 

## Train the model
The model comes in two different versions: As a native PyTorch model class (model.py) and as a PyTorch LightningModule (model_pl.py). Both models are based on the pre-trained Bert model.
Both models have their own training script, train_model.py and train_model_pl.py respectively, that can be run as they are. Since the training data is quite big, it can be recommended considering starting to run with only the validation data for the training and then split that one into a training- and validation dataset. The hyperparameters can be changed in the command-line. You can also choose to optimize the hyperparameters in the PyTorch Lightning training script by adding the argument ``--optimise`` when calling the script. This will cause the script to use Optuna for hyperparameter optimization https://optuna.readthedocs.io/en/stable/index.html. 
While running the train_model_pl.py file the train loss for each batch, validation loss for each epoch, validation accuracy for each epoch and the per class accuracy per epoch will be logged together with the used hyperparameters in a tensorboard events file. This is all handled by the pytorch-lightning module automatically. 



Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── azure_script
    │   ├── Run Bert2Punc in Azure.ipynb    <- Script with guide to train the model in Azure
    │   ├── azure_train_model_pl.py     	<- Training script changed to fit Azure
    │   └── conda_dependencies.yml      	<- Dependencies needed to create environment in Azure
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
    │                         generated with the pipreqs package
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset.py				<- Used for downloading and processing data
	│   │   └── load_data.py				<- Used to load datasets for training, testing, etc.
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
	│   │   ├── drift.py					<- Script for testing data drift
    │   │   ├── model.py					<- Definition of the transformers/pytorch model
	│   │   ├── model_pl.py					<- Definition of the transformers/pytorch-lightning model
    │   │   ├── evaluate_model.py			<- Loads and evaluates a model on the test set (unfinished)
	│   │   ├── evaluate_model_pl.py		<- Loads and evaluates a pytorch-lightning model on the
    │   │   │                                  test set
    │   │   ├── train_model.py				<- Training script using native pytorch (unfinished)
	│   │   └── train_model_pl.py			<- Training script using pytorch-lightning
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize_data.py			<- Prints information about data and shows example
    │
    └── tests              <- Testing scripts to be used with pytest (currently does not support PL)


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
