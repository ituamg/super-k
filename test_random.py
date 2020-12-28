import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from super_k.super_k import *
from plot_funcs import plot_boundaries

def main():
    random_state = 6

    X, y = make_classification(n_samples=1000, n_classes=3, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=random_state)
    
    figsize = (4, 5.25)

    spk = SuperK(k=2)
    spk.fit(X, y)
    plot_boundaries(spk, "random_1", figsize)

    spk = SuperK(k=5)
    spk.fit(X, y)
    plot_boundaries(spk, "random_2", figsize)

    spk = SuperK(k=8)
    spk.fit(X, y)
    plot_boundaries(spk, "random_3", figsize)


if __name__ == "__main__":
    main()