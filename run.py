import json
import pandas as pd
from Unbias import Unbias
from nn import *
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter


def initialize_model(source_df, params):
    print(source_df.shape[0], "datapoints")
    source_df = tokenize_data(source_df, "text")
    source_df = remove_empty(source_df, "text")
    print(source_df.shape[0], "datapoints after removing empty strings")
    df_text = source_df["text"].values.tolist()
    vocab = learn_vocab(df_text, params["vocab_size"])
    df_tokens, SGT, count, SGT_dict = tokens_to_ids(df_text, vocab, params["SGT_path"])

    params["num_SGT"] = count
    print(count, "unique SGTs")
    unique = list(set(SGT))
    unique.sort()

    sgt_W = [1 - (Counter(SGT)[i] / len(SGT)) for i in unique]
    hate_W = [1 - (Counter(source_df["hate"])[i] / source_df.shape[0]) for i in [0, 1]]
    off_W = [1 - (Counter(source_df["offensive"])[i] / source_df.shape[0]) for i in [0, 1]]

    model = Unbias(params, vocab, sgt_W, hate_W, off_W)
    batches = get_batches(df_tokens,
                          model.batch_size,
                          vocab.index("<pad>"),
                          source_df["hate"].tolist(),
                          source_df["offensive"].tolist(),
                          SGT=SGT)
    model.train(batches)
    return model, vocab, SGT_dict

def test_model(test_df, vocab, model, SGT_dict):
    test_df = tokenize_data(test_df, "text")
    test_df = remove_empty(test_df, "text")
    print(test_df.shape[0], "datapoints in test data")
    test_hate = test_df["hate"].values.tolist()
    test_offensive = test_df["offensive"].values.tolist()
    test_text = test_df["text"].values.tolist()
    test_tokens, SGT, count, _ = tokens_to_ids(test_text, vocab, params["SGT_path"], SGT_dict)
    batches = get_batches(test_tokens,
                          params["batch_size"],
                          vocab.index("<pad>"),
                          SGT=SGT)

    test_predictions = model.predict_hate(batches, ["hate", "offensive"])
    print("Hate: F1 score:", f1_score(test_hate, test_predictions["hate"]),
          ", Precision:", precision_score(test_hate, test_predictions["hate"]),
          ", Recall:", recall_score(test_hate, test_predictions["hate"])
          )
    print(Counter(test_hate))
    print(Counter(test_predictions["hate"]))
    print("Offensive: F1 score:", f1_score(test_offensive, test_predictions["offensive"]),
          ", Precision:", precision_score(test_offensive, test_predictions["offensive"]),
          ", Recall:", recall_score(test_offensive, test_predictions["offensive"])
          )
    print(Counter(test_offensive))
    print(Counter(test_predictions["offensive"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to data; includes text, hate and offensive columns")
    parser.add_argument("--params", help="Parameter files. should be a json file")
    parser.add_argument("--test",)

    args = parser.parse_args()

    data = pd.read_csv(args.data)
    test = pd.read_csv(args.test)
    try:
        with open(args.params, 'r') as fo:
            params = json.load(fo)
    except Exception:
        print("Wrong params file")
        exit(1)
    model, vocab, SGT = initialize_model(data, params)
    test_model(test, vocab, model, SGT)


