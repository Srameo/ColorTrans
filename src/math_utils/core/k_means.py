from sklearn.cluster import KMeans
from src.image_control.core.control import ImageController


class KMeansUtil:
    def __init__(self, num):
        self.n_clusters = num
        self.res = None
        self.label_res = None
        self.core = KMeans(n_clusters=num)

    def fit(self, img: ImageController):
        h, w, c = img.ndarray.shape
        x = img.ndarray.reshape(h * w, c)
        self.core.fit(x)
        label = self.core.labels_  # 每个样本的标签
        # 获取聚类中心，并将其铺平
        centers = self.core.cluster_centers_
        new_img = centers[label].reshape((h, w, c))
        self.res = ImageController(matrix=new_img, clr=img.clr)

    def label(self):
        return self.core.labels_

    def centers(self):
        return self.core.cluster_centers_
