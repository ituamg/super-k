import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from super_k.super_k import *
from plot_funcs import plot_boundaries

def main():
    random_state = 111

    X, y = make_moons(n_samples=1000, random_state=random_state, noise=0.15)

    figsize = (4, 2.75)

    spk = SuperK(k=3)
    spk.fit(X, y)
    plot_boundaries(spk, "moons_1", figsize)

    spk = SuperK(k=10)
    spk.fit(X, y)
    plot_boundaries(spk, "moons_2", figsize)

    spk = SuperK(k=17)
    spk.fit(X, y)
    plot_boundaries(spk, "moons_3", figsize)


if __name__ == "__main__":
    main()