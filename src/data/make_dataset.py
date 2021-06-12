from variables import PROJECT_PATH
import datasets
from transformers import AutoTokenizer

def download_data(path=str(PROJECT_PATH / 'data' / 'raw')):
    data = datasets.load_dataset('wikipedia', '20200501.en')
    data.save_to_disk(path)
    return data

def load_data(path=str(PROJECT_PATH / 'data' / 'raw')):
    return datasets.load_from_disk(path)

def tokenize_function(datapoint):
    return tokenizer(datapoint["text"], padding="max_length", truncation=True)

if __name__ == '__main__':
    try:
        data = load_data()
    except:
        data = download_data()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    processed_data = data.map(tokenize_function, batched=True)
    processed_data.save_to_disk(str(PROJECT_PATH / 'data' / 'processed'))