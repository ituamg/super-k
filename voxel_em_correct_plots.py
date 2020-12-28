import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from super_k.super_k import *
from plot_funcs import plot_boundaries

def main():
    random_state = 111

    X, y = make_circles(n_samples=1000, random_state=random_state, noise=0.07)

    window = (X > 0).all(axis=1)
    X_window = X[window]
    y_window = y[window]

    spk = SuperK(k=5)

    spk.load_data(X_window, y_window)

    figsize = (5, 5)

    vox_genpts = spk.voxelize()
    spk.genpts = np.vstack([class_genpts for _, class_genpts in vox_genpts])
    spk.genpts_labels = np.hstack([np.full(len(class_genpts), class_inx) for class_inx, class_genpts in vox_genpts])
    plot_boundaries(spk, "voxelized", figsize)

    em_genpts = spk.apply_em(vox_genpts)
    spk.genpts = np.vstack([class_genpts for _, class_genpts in em_genpts])
    spk.genpts_labels = np.hstack([np.full(len(class_genpts), class_inx) for class_inx, class_genpts in em_genpts])
    plot_boundaries(spk, "after_em", figsize)

    spk.genpts, spk.genpts_labels = spk.merge_and_label(em_genpts)
    plot_boundaries(spk, "relabeled", figsize)

    spk.correct()
    plot_boundaries(spk, "corrected", figsize)



if __name__ == "__main__":
    main()