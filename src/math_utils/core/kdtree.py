from sklearn.neighbors import KDTree
import numpy as np


class KDTreeUtil:
    def __init__(self, data, leaf_size=40):
        self.core = KDTree(np.concatenate(data, axis=0), leaf_size)

    def query(self, dot: np.ndarray, k=1, return_dist=False):
        return self.core.query(dot, k=k, return_distance=return_dist)