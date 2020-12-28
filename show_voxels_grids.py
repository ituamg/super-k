import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_circles
from super_k.super_k import *
from matplotlib import cm
from matplotlib.ticker import StrMethodFormatter


def plot_voxels(ax, X, genpts, title, indices, norm):
    # ax.set_title(title)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ax.scatter(X[:, 0], X[:, 1], cmap="tab20", norm=norm, c=indices)
    # Generator points
    ax.scatter(genpts[:,0], genpts[:,1], c=np.arange(genpts.shape[0]), s=180)
    ax.scatter(genpts[:,0], genpts[:,1], c="w", s=150)
    for i in range(genpts.shape[0]):
        t = ax.text(*genpts[i], str(i), ha="center", va="center", size=9, bbox=dict(boxstyle="circle,pad=0.3", fc="cyan", ec="b", lw=2))


def voxelize_and_plot(X, y, k, fig_name):
    norm = colors.Normalize(0, 19)
    # x_limits = np.array([X.min(axis=0)[0] - 0.05, X.max(axis=0)[0] + 0.05])
    # y_limits = np.array([X.min(axis=0)[1] - 0.05, X.max(axis=0)[1] + 0.05]) 
    color_increment = 0
    for inx in np.unique(y):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_aspect('equal')
        means, n_steps = voxelize(X[y == inx], k)
        X_q, resolution, x_min = quantize(X[y == inx], n_steps)
        bins, indices = np.unique(X_q, axis=0, return_inverse=True)
        plot_voxels(ax, X[y == inx], means, "class {}".format(inx), indices + color_increment, norm)
        color_increment = len(bins)
        ax.grid(True)
        grid_min = X[y == inx].min(axis=0)
        grid_max = X[y == inx].max(axis=0)
        x_grid = np.linspace(grid_min[0], grid_max[0], n_steps[0] + 1)
        y_grid = np.linspace(grid_min[1], grid_max[1], n_steps[1] + 1)
        ax.set_xticks(x_grid)
        ax.set_yticks(y_grid)
        # ax.set_xlim(x_limits)
        # ax.set_ylim(y_limits)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.99, bottom=0.01)
        fig.savefig("images/{}_{}.pdf".format(fig_name, inx))


def main():
    
    random_state = 111

    X, y = make_circles(n_samples=1000, random_state=random_state, noise=0.07)
    window = (X > 0).all(axis=1)

    X_win = X[window]
    y_win = y[window]

    k = 5
    
    voxelize_and_plot(X_win, y_win, k, "voxelization")


if __name__ == "__main__":
    main()