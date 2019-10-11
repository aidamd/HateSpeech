import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot():
    for a in ["losses", "SGT", "hate", "offensive"]:
        losses = pd.read_csv(a + ".csv")
        x = np.linspace(0, losses.shape[0], losses.shape[0])
        plt.plot(x, losses["test"], color = "red")
        plt.plot(x, losses["train"], color = "blue")
        plt.savefig(a)

    losses = pd.read_csv("losses.csv")
    x = np.linspace(0, losses.shape[0], losses.shape[0])
    plt.plot(x, losses["SGT"], color="red")
    plt.plot(x, losses["offensive"], color="blue")
    plt.plot(x, losses["hate"], color="purple")
    plt.savefig("losses")

