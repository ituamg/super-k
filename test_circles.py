import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from super_k.super_k import *
from plot_funcs import plot_boundaries

def main():
    random_state = 111

    X, y = make_circles(n_samples=1000, random_state=random_state, noise=0.05)
    
    figsize = (4, 4.5)

    spk = SuperK(k=10)
    spk.fit(X, y)
    plot_boundaries(spk, "circles_1", figsize)

    spk = SuperK(k=20)
    spk.fit(X, y)
    plot_boundaries(spk, "circles_2", figsize)

    spk = SuperK(k=30)
    spk.fit(X, y)
    plot_boundaries(spk, "circles_3", figsize)


if __name__ == "__main__":
    main()