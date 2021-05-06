import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from umap import UMAP

# GENERAL PARAMETERS
RANDOM_SEED = 42
MULTIPROCESSING_POOL_SIZE = 40
GRID = np.mgrid[0:1:30j, 0:1:30j].reshape(2, -1).T

# DIMENSIONALITY AUGMENTATION PARAMETERS
TARGET_DIMENSIONALITY = 7
RANDOM_NOISE_STD = 0.01
USED_CURVATURES = ("logistic", "polynomial_roll", "sine", "circle", "flat")
CURVATURES = {
    "flat": lambda x: lambda t: 0,
    "circle": lambda x: lambda t: x * np.pi * 2,
    "polynomial_roll": lambda x: lambda t: x * 4 * (t + 1) ** (x * 2),
    "roll": lambda x: lambda t: np.exp(x * t * 4),
    "gaussian": lambda x: lambda t: 1 / (x * 0.1) * np.exp(-(t ** 2) / (x * 0.1) ** 2),
    "sine": lambda x: lambda t: (5 + (x - 1) * 10) * np.sin(t * np.pi * 2),
    "logistic": lambda x: lambda t: (x * 10) / (1 + np.exp(-0.5 * t)),
    "relu": lambda x: lambda t: 0 if t < 0 else x * t * 10,
}

# DIMENSIONALITY REDUCTION PARAMETERS
DIFFICULTY = (0.8, 1.2)
DIMENSIONALITY_REDUCTION_MODELS = (
    PCA(n_components=2, random_state=RANDOM_SEED),
    Isomap(n_components=2, eigen_solver="dense"),
    TruncatedSVD(n_components=2, random_state=RANDOM_SEED),
    TSNE(n_components=2, random_state=RANDOM_SEED),
    UMAP(n_components=2, random_state=RANDOM_SEED, metric="cosine"),
)
