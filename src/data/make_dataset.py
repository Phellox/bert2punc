import string
from variables import PROJECT_PATH
import datasets
import torch
from transformers import BertTokenizer


def download_data(path=str(PROJECT_PATH / "data" / "raw")):
    data = datasets.load_dataset("wikipedia", "20200501.en")
    data.save_to_disk(path)
    return data


def load_data(path=str(PROJECT_PATH / "data" / "raw")):
    return datasets.load_from_disk(path)


def encode_data(data, tokenizer, punctuation_enc, segment_size):
    """
    Converts words to (BERT) tokens and punctuation to given encoding.
    Note that words can be composed of multiple tokens.
    """

    X = []
    Y = []
    for idx, split_text in enumerate(data):
        if len(split_text) >= segment_size:
            X_tmp = []
            for word, punc in split_text:
                tokens = tokenizer.tokenize(word)
                x = tokenizer.convert_tokens_to_ids(tokens)
                y = [punctuation_enc[punc]]
                if len(x) > 0:
                    if len(x) > 1:
                        y = (len(x)-1)*[0] + y
                    X_tmp += x
                    Y += y
            X.append(X_tmp)

    return X, torch.tensor(Y)


def insert_target(X, segment_size):
    """
    Creates segments of surrounding words for each word in x.
    Inserts a zero token halfway the segment.
    """
    X_flattened = []

    n = 0
    for x in X:
        x_pad = x[-((segment_size-1)//2-1):]+x+x[:segment_size//2]

        for i in range(len(x_pad)-segment_size+2):
            segment = x_pad[i:i+segment_size-1]
            segment.insert((segment_size-1)//2, 0)
            X_flattened.append(segment)

        n += len(x)
        if n != len(X_flattened):
            print("ERROR: DIMENSION MISMATCH")

    return torch.tensor(X_flattened)


def reformat_data(data, include_punctuations=".,"):
    exclude_punctuations = string.punctuation.translate(str.maketrans('', '', include_punctuations))
    table_exclude_specific = str.maketrans('', '', exclude_punctuations)
    table_exclude_all = str.maketrans('', '', string.punctuation)

    data = [text.translate(table_exclude_specific).split() for text in data]
    for idx, split_text in enumerate(data):
        tmp_split_text = [(word, word[-1]) if word[-1] in include_punctuations else (word, ' ') for word in split_text]
        tmp_split_text = [(word.translate(table_exclude_all), punc) for word, punc in tmp_split_text if word != ""]
        data[idx] = tmp_split_text

    return data


def preprocess_data(data, tokenizer, punctuation_enc, segment_size):
    include_punctuations = ("".join(punctuation_enc.keys())).replace(" ", "")
    data = reformat_data(data, include_punctuations)
    X, Y = encode_data(data, tokenizer, punctuation_enc, segment_size)
    X = insert_target(X, segment_size)
    return X, Y


if __name__ == "__main__":
    # Load data if possible, otherwise download it (lazy method)
    try:
        data = load_data()
    except:
        data = download_data()

    # Encode punctuations of interest
    punctuation_enc = {
        ' ': 1,
        '.': 2,
        ',': 3
    }

    # Define tokenizer for bert-base-uncased and segment size for each observation
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    segment_size = 32

    # Print number of articles
    print("Total number of articles: {}".format(len(data["train"])))

    # Trim data for faster computations
    train_data = data["train"][:100]["text"]
    val_data = data["train"][100:110]["text"]
    test_data = data["train"][110:120]["text"]

    # Create processed data
    print("Begin preprocessing of data")
    X_train, y_train = preprocess_data(train_data, tokenizer, punctuation_enc, segment_size)
    X_val, y_val = preprocess_data(val_data, tokenizer, punctuation_enc, segment_size)
    X_test, y_test = preprocess_data(test_data, tokenizer, punctuation_enc, segment_size)
    print("Done!")

    # Save data to pytorch files (this assumes the data set is fairly small s.t. all data can be stored in memory)
    print("Saving processed data to: {}".format(str(PROJECT_PATH / "data" / "processed")))
    torch.save((X_train, y_train), PROJECT_PATH / "data" / "processed" / "train.pt")
    torch.save((X_val, y_val), PROJECT_PATH / "data" / "processed" / "val.pt")
    torch.save((X_test, y_test), PROJECT_PATH / "data" / "processed" / "test.pt")
    print("Done!")
