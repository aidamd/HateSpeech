import json
import pandas as pd
from Bias import Bias
from nn import *
import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="")
    args = parser.parse_args()

    Bias(args.data, model_path="saved_model/vanilla/hate")

