import pandas as pd
import argparse
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(df.shape)
    mask = np.random.rand(df.shape[0]) < 0.8
    train = df[mask]
    print(train.shape)
    test = df[~mask]
    print(test.shape)

    train.to_csv(os.path.splitext(args.data)[0] + "_train.csv", index=False)
    test.to_csv(os.path.splitext(args.data)[0] + "_test.csv", index=False)
