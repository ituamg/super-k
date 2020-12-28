import numpy as np
import matplotlib.pyplot as plt
from super_k.super_k import maximization

def plot_assignments(ax, X, genpts, title, coloring):

    ax.set_title(title)
    ax.scatter(X[:, 0], X[:, 1], c=coloring)
    # Generator points
    ax.scatter(genpts[:,0], genpts[:,1], c=np.arange(genpts.shape[0]), s=180)
    ax.scatter(genpts[:,0], genpts[:,1], c="w", s=150)
    for i in range(genpts.shape[0]):
        t = ax.text(*genpts[i], str(i), ha="center", va="center", size=9, bbox=dict(boxstyle="circle,pad=0.3", fc="cyan", ec="b", lw=2))

def plot_boundaries(spk, name, figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    assignments = maximization(spk.samples, spk.genpts)

    plot_assignments(ax, spk.samples, spk.genpts, "k = {}, n_genpts = {}, accuracy = {:.3f}".format(spk.k, spk.n_genpts, spk.rate()), spk.genpts_labels[assignments])

    # False assignments
    X_false = spk.samples[(spk.labels != spk.genpts_labels[assignments])]
    ax.scatter(X_false[:, 0], X_false[:, 1], c="r", s=5)
        
    # Draw boundaries of Classes using contour plot
    x = np.linspace(*ax.get_xlim(), 1000)
    y = np.linspace(*ax.get_ylim(), 1000)
    xx, yy = np.meshgrid(x, y)
    xy = np.c_[np.ravel(xx), np.ravel(yy)]
    
    class_indices = spk.predict(xy)
    class_indices = class_indices.reshape(xx.shape)
    ax.contour(xx, yy, class_indices, colors="magenta", linewidths=2.0)

    # fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.04)
    fig.tight_layout()
    fig.savefig("images/{}.pdf".format(name))