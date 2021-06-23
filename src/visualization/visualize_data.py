from src.data.make_dataset import load_data, encode_data, insert_target, reformat_data
from src.data.load_data import load_dataset
from transformers import BertTokenizer

if __name__ == "__main__":
    # Load data and show raw data example as well as data sizes
    data = load_data()
    print("TOTAL NUMBER OF ARTICLES:")
    print(len(data["train"]))
    print("")

    print("NUMBER OF ARTICLES IN EACH DATASET:")
    print("train: 100")
    print("val:    10")
    print("test:   10")
    print("")

    train_data = data["train"][:100]["text"]
    train_data_example = train_data[3]
    print("RAW TRAIN DATA EXAMPLE:")
    print(train_data_example)
    print("")

    # Show which punctuations will be used
    punctuation_enc = {
        ' ': 0,
        '.': 1,
        ',': 2
    }
    include_punctuations = ("".join(punctuation_enc.keys())).replace(" ", "")
    print("PUNCTUATIONS OF INTEREST:")
    print(include_punctuations)
    print("")

    # Determine and show reformatted data
    train_data_example = reformat_data([train_data_example], include_punctuations)
    print("REFORMATTED TRAIN DATA EXAMPLE:")
    for text in train_data_example:
        for word_and_punc in text[:9]:
            print(word_and_punc)
    print("")

    # Encode data and show the encoded data
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    segment_size = 32
    X, Y = encode_data(train_data_example, tokenizer, punctuation_enc, segment_size)
    print("ENCODED/TOKENIZED TEXT EXCERPT:")
    print(X[0][:11])
    print("")
    print("LABELS EXCERPT")
    print(Y[:11])
    print("")

    # Create segments of text with inserted target
    X = insert_target(X, segment_size)
    print("ENCODED/TOKENIZED SEGMENT WITH TARGET EXAMPLE:")
    print(X[0])
    print("")

    # Print number of segments in each dataset
    print("NUMBER OF SEGMENTS IN EACH DATASET:")
    print("train: {}".format(len(load_dataset("train"))))
    print("val: {}".format(len(load_dataset("val"))))
    print("test: {}".format(len(load_dataset("test"))))
    print("")



