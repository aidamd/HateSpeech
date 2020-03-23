import json
import pandas as pd
from Unbias import Unbias
from nn import *
import argparse

from collections import Counter


def initialize_model(train_df, params):
    train_df = preprocess(train_df)

    train_text = train_df["text"].values.tolist()
    vocab = learn_vocab(train_text, params["vocab_size"])

    train_tokens, SGT, count, SGT_dict = tokens_to_ids(train_text, vocab, params["SGT_path"])

    params["num_SGT"] = count
    print(count, "unique SGTs")
    unique = list(set(SGT))
    unique.sort()

    sgt_W = cal_weights(SGT)
    hate_W = cal_weights(train_df["hate"].tolist())
    off_W = cal_weights(train_df["offensive"].tolist())

    model = Unbias(params, vocab, sgt_W, hate_W, off_W)
    batches = get_batches(train_tokens,
                          model.batch_size,
                          vocab.index("<pad>"),
                          train_df["hate"].tolist(),
                          train_df["offensive"].tolist(),
                          SGT=SGT)
    model.train(batches)
    return model, vocab, SGT_dict

def test_model(test_df, vocab, model, SGT_dict):
    test_df = preprocess(test_df)
    test_text = test_df["text"].values.tolist()

    test_tokens, SGT, count, _ = tokens_to_ids(test_text, vocab, params["SGT_path"], SGT_dict)
    batches = get_batches(test_tokens,
                          params["batch_size"],
                          vocab.index("<pad>"),
                          SGT=SGT)

    test_predictions = model.predict_hate(batches, ["hate", "offensive"])
    prediction_results(test_df, test_predictions, ["hate", "offensive"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to data; includes text, hate and offensive columns")
    parser.add_argument("--params", help="Parameter files. should be a json file")
    parser.add_argument("--test",)

    args = parser.parse_args()

    data = pd.read_csv(args.data)
    test = pd.read_csv(args.test)
    try:
        params = json.load(open(args.params, 'r'))
    except Exception:
        print("Wrong params file")
        exit(1)

    model, vocab, SGT = initialize_model(data, params)
    test_model(test, vocab, model, SGT)


