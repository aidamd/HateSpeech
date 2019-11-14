import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot():
    for a in ["SGT", "hate", "offensive"]:
        losses = pd.read_csv("plots/" + a + ".csv")
        x = np.linspace(0, losses.shape[0], losses.shape[0])
        plt.figure()
        plt.plot(x, losses["test"], color = "red")
        plt.plot(x, losses["train"], color = "blue")
        plt.legend()
        plt.savefig("plots/" + a)

    plt.figure()
    losses = pd.read_csv("plots/" + "losses.csv")
    x = np.linspace(0, losses.shape[0], losses.shape[0])
    plt.plot(x, losses["SGT"], color="red")
    plt.plot(x, losses["offensive"], color="blue")
    plt.plot(x, losses["hate"], color="#DAA520")
    plt.legend()
    plt.savefig("plots/losses")


if __name__ == "__main__":
    plot()