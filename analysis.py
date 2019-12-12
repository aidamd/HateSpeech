import pandas as pd
from collections import Counter
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()

    fake = pd.read_csv(args.data + "/predictions.csv")

    orig = pd.read_csv("Data/24k/gab.csv")
    samples = set(fake["sample"].tolist())

    # the changes that changes a non hate text to hate
    fp = list()
    fp_change = list()

    # the changes that changes a hate text to non hate
    fn = list()
    fn_change = list()

    """
    for s in samples:
        subset = fake.loc[fake["sample"] == s]
        if len(set(subset["predicted_hate"])) > 1:
            print("original text = " + orig.iloc[s, 4])
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
    """
    for i, row in fake.iterrows():
        if row["hate"] != row["predicted_hate"]:
            if row["predicted_hate"] == 1:
                fp.append(row["origin_SGT"] + "->" + row["SGT"])
                fp_change.append(row["SGT"])
            else:
                fn.append(row["origin_SGT"] + "->" + row["SGT"])
                fn_change.append(row["SGT"])

    print("Changes that made the text hateful")
    print(Counter(fp))
    pd.DataFrame.from_dict({"Change": list(Counter(fp).keys()),
                           "Frequency": list(Counter(fp).values())})\
        .to_csv(args.data + "/became_hate.csv", index=False)
    pd.DataFrame.from_dict({"Change": list(Counter(fp_change).keys()),
                           "Frequency": list(Counter(fp_change).values())})\
        .to_csv(args.data + "/became_hate_SGT.csv", index=False)


    print("Changes that made the text not hateful")
    print(Counter(fn))
    pd.DataFrame.from_dict({"Change": list(Counter(fn).keys()),
                           "Frequency": list(Counter(fn).values())})\
        .to_csv(args.data + "/became_nonhate.csv", index=False)
    pd.DataFrame.from_dict({"Change": list(Counter(fn_change).keys()),
                           "Frequency": list(Counter(fn_change).values())})\
        .to_csv(args.data + "/became_nonhate_SGT.csv", index=False)