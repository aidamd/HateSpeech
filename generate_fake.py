from nn import *
import pandas as pd
import numpy as np

def generate_fake(data_path="Data/24k/gab_train.csv", test_path="Data/24k/gab_test.csv"):
    source_df = pd.read_csv(data_path)
    print(source_df.shape[0], "datapoints")

    source_df = tokenize_data(source_df, "text")
    source_df = remove_empty(source_df, "text")
    print(source_df.shape[0], "datapoints after removing empty strings")
    df_text = source_df["text"].values.tolist()
    vocabs = learn_vocab(df_text, 10000)

    test = pd.read_csv(test_path)
    SGT_test = get_SGTs(test, vocabs).reset_index()
    print("Number of samples containing SGTs in test set:", SGT_test.shape[0])

    pos = SGT_test.loc[SGT_test["hate"] == 1,].index
    print("Number of positive samples containing SGTs in test set:", pos.shape[0])
    neg = SGT_test.loc[SGT_test["hate"] == 0,].index
    balaced = list(pos.values)
    balaced.extend(list(np.random.choice(neg, pos.shape[0], replace=False)))
    print(len(balaced))

    fake = expand(SGT_test.iloc[balaced,:], vocabs)
    fake.to_csv("Data/24k/fake_test.csv", index=False)
    print(fake.shape)

def get_SGTs(corpus, vocabs, SGT_path="extended_SGT.txt"):
    SGT_dare = list()
    SGTs = [tok.replace("\n", "") for tok in open(SGT_path, "r").readlines()]
    SGTs = list(set([tok for tok in SGTs if tok in vocabs]))
    print("Number of SGTs:", len(SGTs))
    for i, row in corpus.iterrows():
        text = nltk_token.WordPunctTokenizer().tokenize(clean(row["text"]))
        for j, tok in enumerate(text):
            for s in SGTs:
                if s in tok:
                    SGT_dare.append(i)
    SGT_dar = corpus.iloc[SGT_dare, :]
    return SGT_dar

def expand(corpus, vocabs, SGT_path="extended_SGT.txt"):
    SGTs = read_SGT(vocabs, SGT_path).keys()
    fake_df = {"text": list(),
               "hate": list(),
               "sample": list(),
               "SGT": list(),
               "origin_SGT": list()}
    print(corpus.shape)
    for i, row in corpus.iterrows():
        clean_text = clean(row["text"])
        text = nltk_token.WordPunctTokenizer().tokenize(clean_text)
        for s in SGTs:
            if s in text:
                fake_df["text"].extend([clean_text.replace(s, _sgt) for _sgt in SGTs])
                fake_df["hate"].extend([row["hate"] for t in SGTs])
                fake_df["sample"].extend([i for t in SGTs])
                fake_df["SGT"].extend([t for t in SGTs])
                fake_df["origin_SGT"].extend([s for t in SGTs])
                break


    return pd.DataFrame.from_dict(fake_df)

if __name__ == '__main__':
    generate_fake()
