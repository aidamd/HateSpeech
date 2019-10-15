import pandas as pd
from collections import Counter
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()

    fake = pd.read_csv(args.data + "/predictions.csv")

    orig = pd.read_csv("Data/gab.csv")
    samples = set(fake["sample"].tolist())

    # the changes that changes a non hate text to hate
    hn = list()
    hn_change = list()

    # the changes that changes a hate text to non hate
    nh = list()
    nh_change = list()

    for s in samples:
        subset = fake.loc[fake["sample"] == s]
        if len(set(subset["predicted_hate"])) > 1:
            print("original text = " + orig.iloc[s, 3])
            print("original label = " + "Hate" if orig.iloc[s, 1] == 1 else "Not Hate")
            for i, row in subset.iterrows():
                if row["predicted_hate"] != orig.iloc[s, 1]:
                    print(row["SGT"])
                    if orig.iloc[s, 1] == 1:
                        hn.append(row["origin_SGT"] + "->" + row["SGT"])
                        hn_change.append(row["SGT"])
                    else:
                        nh.append(row["origin_SGT"] + "->" + row["SGT"])
                        nh_change.append(row["SGT"])

    print("Changes that made the text hateful")
    print(Counter(nh))
    pd.DataFrame.from_dict({"Change": list(Counter(nh).keys()),
                           "Frequency": list(Counter(nh).values())})\
        .to_csv(args.data + "/became_hate.csv", index=False)
    pd.DataFrame.from_dict({"Change": list(Counter(nh_change).keys()),
                           "Frequency": list(Counter(nh_change).values())})\
        .to_csv(args.data + "/became_hate_SGT.csv", index=False)


    print("Changes that made the text not hateful")
    print(Counter(hn))
    pd.DataFrame.from_dict({"Change": list(Counter(hn).keys()),
                           "Frequency": list(Counter(hn).values())})\
        .to_csv(args.data + "/became_nonhate.csv", index=False)
    pd.DataFrame.from_dict({"Change": list(Counter(hn_change).keys()),
                           "Frequency": list(Counter(hn_change).values())})\
        .to_csv(args.data + "/became_nonhate_SGT.csv", index=False)