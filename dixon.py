from Bias import Bias
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="")
    args = parser.parse_args()

    Bias(args.data, model_path="saved_model/dixon/hate", oversample=True)

