import numpy as np
from math import ceil, floor, log

from numpy.core.numeric import Inf


def likelihood(X: np.ndarray, M: np.ndarray):
    """Simplified multivariate lognormal likelihood

    Simplified for Σ=I and equal priors

    Args:
        X (np.ndarray): Samples
        M (np.ndarray): Target points

    Returns:
        (np.ndarray) : Likelihood
    """    

    return X @ M.T - 0.5 * (M**2).sum(axis=1)


def quantize(X, n_steps, eps=1e-15):
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    resolution = (x_max - x_min) / n_steps
    X_q = np.clip((X - x_min) / (resolution + eps), None, n_steps - 1).astype(int)
    X_q[:,resolution == 0] = 0 # correction for NaNs happening because of 0 / 0
    return X_q, resolution, x_min


def voxelize(X, k):
    N, m = X.shape

    #### Voxelization parameters ####
    c = k ** (1 / m) # k = c^m => c = ?
    a, b = floor(c), ceil(c) # integer range of c
    m_v = round(m * log(c / a, b / a)) # Number of variant features

    max_steps = np.array([len(np.unique(X[:, inx])) for inx in range(m)])
    n_steps = np.full(m, a)
    
    if m_v > 0:
        variants = np.sort(max_steps.argpartition(m - m_v)[m - m_v:])
        n_steps[variants] = b
    
    invalid_steps = n_steps > max_steps
    if invalid_steps.any():
        n_steps[invalid_steps] = max_steps[invalid_steps]

    #### Voxelization through quantization ####
    X_q, _, _ = quantize(X, n_steps)
    bins, indices, counts = np.unique(X_q, axis=0, return_inverse=True, return_counts=True)
    voxel_means = np.array([X[indices == inx].mean(axis=0) for inx in range(len(bins))])

    return voxel_means, n_steps


def maximization(X, means):
    return likelihood(X, means).argmax(axis=1)


def expectation(X, assignments):
    return np.array([X[assignments == inx].mean(axis=0) for inx in np.unique(assignments)])


def cycle_through_em(X, means, n_cycles=1, r_delta=0.01):
    n_delta = int(len(X) * r_delta)
    old_assignments = None
    for _ in range(n_cycles):
        assignments = maximization(X, means)
        means = np.array([X[assignments == inx].mean(axis=0) for inx in np.unique(assignments)])
        if old_assignments is not None:
            if (assignments != old_assignments).sum() <= n_delta:
                break
        old_assignments = assignments
    return means


class SuperK:
    """Super-k Algorithm
    """

    RATE_FORMAT = "12.10f"

    def __init__(self, k=2, dtype=np.float32, **kwargs):
        self.k = k
        self.dtype = dtype

    ##########################################
    # For scikit-learn estimator compatibility
    def get_params(self, deep=True):
        return {"k": self.k, "dtype": self.dtype}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    ##########################################

    def load_data(self, X, y):
        self.samples = X.astype(self.dtype)
        self.classes, self.labels = np.unique(y, return_inverse=True)
        self.n_classes = self.classes.shape[0]

    @property
    def n_samples(self): return self.samples.shape[0]

    @property
    def n_dim(self): return self.samples.shape[1]

    @property
    def n_genpts(self): return self.genpts.shape[0]

    def voxelize(self):
        vox_genpts = []

        for cls_inx in range(self.n_classes):
            cls_mask = self.labels == cls_inx
            class_genpts, _ = voxelize(self.samples[cls_mask], self.k)
            vox_genpts.append((cls_inx, class_genpts))

        return vox_genpts

    def apply_em(self, genpts, em_max_cycles=100, em_r_delta=0.01):
        em_genpts = []

        for cls_inx, class_genpts in genpts:
            cls_mask = self.labels == cls_inx
            class_genpts = cycle_through_em(self.samples[cls_mask], class_genpts, em_max_cycles, em_r_delta)
            em_genpts.append((cls_inx, class_genpts))

        return em_genpts

    def merge_and_label(self, genpts):
        merged_genpts = np.vstack([class_genpts for _, class_genpts in genpts])
        assignments = maximization(self.samples, merged_genpts)
        genpts_labels = []

        for inx in range(len(merged_genpts)):
            classes, counts = np.unique(self.labels[assignments == inx], return_counts=True)
            genpts_labels.append(classes[counts.argmax()])

        return merged_genpts, np.array(genpts_labels, dtype=int)

    def correct(self, n_cycles=100, error_bound=2.0):

        best_params = (self.genpts.copy(), self.genpts_labels.copy())
        lowest_error = Inf

        for _ in range(n_cycles):
            assignments = maximization(self.samples, self.genpts)
            classification = self.genpts_labels[assignments]

            fp_mask = classification != self.labels
            fp_samples = self.samples[fp_mask]
            fp_assignments = assignments[fp_mask]

            error = np.sum(fp_mask) / self.n_samples

            if error < lowest_error:
                best_params = (self.genpts.copy(), self.genpts_labels.copy())
                lowest_error = error

            if error > error_bound * lowest_error:
                break

            for genpt_inx in np.unique(fp_assignments):
                n_all = (assignments == genpt_inx).sum()
                n_fp = (fp_assignments == genpt_inx).sum()
                if n_all > n_fp:
                    self.genpts[genpt_inx] = (self.genpts[genpt_inx] * n_all - fp_samples[fp_assignments == genpt_inx].sum(axis=0)) / (n_all - n_fp)
                
        self.genpts, self.genpts_labels = best_params


    def train(self, em_max_cycles=100, em_r_delta=0.01, corr_max_cycles=100):

        #### Voxelize & EM & Merge ####
        genpts = np.empty((0, self.n_dim), dtype=self.dtype)

        for cls_inx in range(self.n_classes):
            cls_mask = self.labels == cls_inx
            class_genpts, _ = voxelize(self.samples[cls_mask], self.k)
            class_genpts = cycle_through_em(self.samples[cls_mask], class_genpts, em_max_cycles, em_r_delta)
            genpts = np.vstack((genpts, class_genpts))

        #### Label ####
        interclass_assignments = maximization(self.samples, genpts)
        genpts_labels = np.empty(len(genpts), dtype=int)

        valid_genpt_indices = np.unique(interclass_assignments)
        for inx, labels in [(inx, self.labels[interclass_assignments == inx]) for inx in valid_genpt_indices]:
            classes, counts = np.unique(labels, return_counts=True)
            genpts_labels[inx] = classes[counts.argmax()]
        self.genpts = genpts[valid_genpt_indices]
        self.genpts_labels = genpts_labels[valid_genpt_indices]

        #### Correct ####
        self.correct(corr_max_cycles)

    def train_step_by_step(self, em_max_cycles=100, em_r_delta=0.01, corr_max_cycles=100):

        vox_genpts = self.voxelize()
        self.genpts = np.vstack([class_genpts for _, class_genpts in vox_genpts])
        self.genpts_labels = np.hstack([np.full(len(class_genpts), class_inx) for class_inx, class_genpts in vox_genpts])

        em_genpts = self.apply_em(vox_genpts, em_max_cycles, em_r_delta)
        self.genpts = np.vstack([class_genpts for _, class_genpts in em_genpts])
        self.genpts_labels = np.hstack([np.full(len(class_genpts), class_inx) for class_inx, class_genpts in em_genpts])

        self.genpts, self.genpts_labels = self.merge_and_label(em_genpts)

        self.correct(corr_max_cycles)

    def fit(self, X, y, verbose=False):
        self.load_data(X, y)
        self.train()
        if verbose:
            print("Number of generator points: {}".format(self.n_genpts))
            print("Rate: {:{rfmt}}".format(self.rate(), rfmt=self.RATE_FORMAT))
        return self

    def predict(self, X):
        genpt_indices = maximization(X.astype(self.dtype), self.genpts)
        class_indices = self.genpts_labels[genpt_indices]
        return self.classes[class_indices]

    def score(self, X, y):
        n_correct = (y == self.predict(X)).sum()
        return n_correct / y.shape[0]

    def rate(self):
        assignments = maximization(self.samples, self.genpts)
        classification = self.genpts_labels[assignments]
        return (classification == self.labels).sum() / self.n_samples
