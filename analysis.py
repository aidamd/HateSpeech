import pandas as pd
from collections import Counter
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()

    fake = pd.read_csv(args.data + "/predictions.csv")

    orig = pd.read_csv("Data/28k/posts_test.csv")
    samples = set(fake["sample"].tolist())

    # the changes that changes a non hate text to hate
    fp = list()
    fp_change = list()

    # the changes that changes a hate text to non hate
    fn = list()
    fn_change = list()

    for i, row in fake.iterrows():
        if row["hate"] != row["predicted_hate"]:
            if row["predicted_hate"] == 1:
                fp.append(row["origin_SGT"] + "->" + row["SGT"])
                fp_change.append(row["SGT"])
            else:
                fn.append(row["origin_SGT"] + "->" + row["SGT"])
                fn_change.append(row["SGT"])

    cases = Counter(fake["SGT"])
    print("Changes that made the text hateful")
    print(Counter(fp))

    #pd.DataFrame.from_dict({"Change": list(Counter(fp_change).keys()),
    #                       "Frequency": [y / cases[x] for x, y in Counter(fp_change).items()]})\
    #    .to_csv(args.data + "/fp_SGT.csv", index=False)
    pd.DataFrame.from_dict({"Change": list(Counter(fp_change).keys()),
                           "Frequency": list(Counter(fp_change).values())})\
        .to_csv(args.data + "/fp_SGT.csv", index=False)
    pd.DataFrame.from_dict({"From": [x.split("->")[0] for x in list(Counter(fp).keys())],
                            "To": [x.split("->")[1] for x in list(Counter(fp).keys())],
                            "Frequency": list(Counter(fp).values())}) \
        .to_csv(args.data + "/fp_change.csv", index=False)


    print("Changes that made the text not hateful")
    print(Counter(fn))
    #pd.DataFrame.from_dict({"Change": list(Counter(fn_change).keys()),
    #                       "Frequency": [y / cases[x] for x, y in Counter(fn_change).items()]})\
    #    .to_csv(args.data + "/fn_SGT.csv", index=False)
    pd.DataFrame.from_dict({"Change": list(Counter(fn_change).keys()),
                            "Frequency": list(Counter(fn_change).values())}) \
        .to_csv(args.data + "/fn_SGT.csv", index=False)
    pd.DataFrame.from_dict({"From": [x.split("->")[0] for x in list(Counter(fn).keys())],
                            "To": [x.split("->")[1] for x in list(Counter(fn).keys())],
                            "Frequency": list(Counter(fn).values())}) \
        .to_csv(args.data + "/fn_change.csv", index=False)