import json
import pandas as pd
from Unbias import Unbias
from nn import *
import argparse
from collections import Counter



def check_bias(source_df, params):
    print(source_df.shape[0], "datapoints")
    source_df = tokenize_data(source_df, "text")
    source_df = remove_empty(source_df, "text")
    print(source_df.shape[0], "datapoints after removing empty strings")
    df_text = source_df["text"].values.tolist()
    vocab = learn_vocab(df_text, params["vocab_size"])
    df_tokens, SGT, count, SGT_dict = tokens_to_ids(df_text, vocab, params["SGT_path"])
    # SGT, count = extract_SGT(df_tokens, vocab, params["SGT_path"])
    #print(SGT_dict)
    params["num_SGT"] = count
    unique = list(set(SGT))
    unique.sort()
    SGT_weights = [1 - (Counter(SGT)[i] / len(SGT)) for i in unique]

    model = Unbias(params, vocab, SGT_weights)

    fake_df = pd.read_csv("Data/24k/fake_test.csv")
    fake_df = tokenize_data(fake_df, "text")
    fake_df = remove_empty(fake_df, "text")
    print(fake_df.shape[0], "datapoints in fake data")
    fake_text = fake_df["text"].values.tolist()
    fake_tokens, SGT, count, _ = tokens_to_ids(fake_text, vocab, params["SGT_path"], SGT_dict)

    batches = get_batches(fake_tokens,
                          params["batch_size"],
                          vocab.index("<pad>"),
                          SGT=SGT)

    predictions = model.predict_hate(batches, ["hate"])["hate"]

    fake_df["predicted_hate"] = predictions
    fake_df.to_csv("unbiased/predictions.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to data; includes text, hate and offensive columns")
    parser.add_argument("--params", help="Parameter files. should be a json file")

    args = parser.parse_args()

    data = pd.read_csv(args.data)

    try:
        with open(args.params, 'r') as fo:
            params = json.load(fo)
    except Exception:
        print("Wrong params file")
        exit(1)
    check_bias(data, params)

