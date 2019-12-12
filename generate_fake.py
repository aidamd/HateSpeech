from nn import *
import pandas as pd
import os

def generate_fake(data_path="Data/24k/gab_test.csv"):
    source_df = pd.read_csv(data_path)
    print(source_df.shape[0], "datapoints")

    source_df = tokenize_data(source_df, "text")
    source_df = remove_empty(source_df, "text")
    print(source_df.shape[0], "datapoints after removing empty strings")
    df_text = source_df["text"].values.tolist()
    vocabs = learn_vocab(df_text, 10000)

    fake = expand(source_df, vocabs)
    fake_path = "Data/24k/fake_gab.csv"
    fake.to_csv(fake_path)
    return fake_path

def expand(corpus, vocabs, SGT_path="extended_SGT.txt"):
    SGTs = [tok.replace("\n", "") for tok in open(SGT_path, "r").readlines()]
    SGTs = [tok for tok in SGTs if tok in vocabs]
    fake_df = {"text": list(),
               "hate": list(),
               "sample": list(),
               "SGT": list(),
               "origin_SGT": list()}
    print(len(SGTs))
    for i, row in corpus.iterrows():
        for j, tok in enumerate(row["text"]):
            for s in SGTs:
                if s in tok:
                    text = " ".join(word for word in row["text"])
                    fake_df["text"].extend([text.replace(tok, _sgt) for _sgt in SGTs])
                    fake_df["hate"].extend([row["hate"] for t in SGTs])
                    fake_df["sample"].extend([i for t in SGTs])
                    fake_df["SGT"].extend([t for t in SGTs])
                    fake_df["origin_SGT"].extend([s for t in SGTs])

    return pd.DataFrame.from_dict(fake_df)

if __name__ == '__main__':
    generate_fake()
