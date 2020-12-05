from sklearn.neighbors import KDTree


class KDTreeUtil:
    def __init__(self, data, leaf_size=40):
        self.core = KDTree(data, leaf_size)