from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="This project aims to use the BERT model to predict where to insert punctuations in sequences of text. Initially, the model will be based on the pretrained BERT model from the transformers repository created by Hugging Face. Similarly, the initial dataset will be english texts from Wikipedia loaded through the transformers repository. If time allows for it, we will also try to use a pretrained Nordic BERT model to predict punctuation in danish texts from Wikipedia.",
    author="BJD, JHT, MES",
    license="MIT",
)
