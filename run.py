import json
import pandas as pd
from Unbias import Unbias
from nn import *
import argparse
import pickle



def oversample(source_df, params):
    print(source_df.shape[0], "datapoints")
    source_df = tokenize_data(source_df, "text")
    source_df = remove_empty(source_df, "text")
    print(source_df.shape[0], "datapoints after removing empty strings")
    df_text = source_df["text"].values.tolist()
    vocab = learn_vocab(df_text, params["vocab_size"])
    df_tokens, SGT, count = tokens_to_ids(df_text, vocab, params["SGT_path"])
    # SGT, count = extract_SGT(df_tokens, vocab, params["SGT_path"])
    params["num_SGT"] = count

    model = Unbias(params, df_tokens, vocab,
                   source_df["hate"].tolist(),
                   source_df["offensive"].tolist(),
                   SGT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to data; includes a text columns and a style comment of 0s and 1s")
    parser.add_argument("--params", help="Parameter files. should be a json file")

    args = parser.parse_args()
    if args.data.endswith('.tsv'):
        data = pd.read_csv(args.data, sep='\t', quoting=3)
    elif args.data.endswith('.csv'):
        data = pd.read_csv(args.data)
    elif args.data.endswith('.pkl'):
        data = pickle.load(open(args.data, 'rb'))

    try:
        with open(args.params, 'r') as fo:
            params = json.load(fo)
    except Exception:
        print("Wrong params file")
        exit(1)
    oversample(data, params)

