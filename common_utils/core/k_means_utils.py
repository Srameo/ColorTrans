import numpy as np
import random


def ndarray_to_list(mat: np.ndarray):
    return [{"color": i, "index": index, "cluster": -1} for index, i in enumerate(mat)]


def dist(a: np.ndarray, b: np.ndarray):
    delta = a - b
    dst = (delta * delta).sum()
    return dst ** 0.5


def update_k_means(mat_lst: list, k: list):
    new_k = [{"sum": np.zeros(i["color"].shape), "count": 0} for i in k]
    dists = []  # 存储一个点到所有聚类的距离
    for dot in mat_lst:
        d_color = dot["color"]
        for c in k:
            c_color = c["color"]
            dists.append(dist(c_color, d_color))
        new_cluster = dists.index(min(dists))
        # 更新每个点的坐标
        dot["cluster"] = new_cluster
        dists.clear()
        # 记录每个聚类的颜色和
        new_k[new_cluster]["sum"] += dot["color"]
        new_k[new_cluster]["count"] += 1
        print(dot)
    # 更新每个聚类的坐标
    for index, cluster in enumerate(k):
        cluster["color"] = new_k[index]["sum"] / new_k[index]["count"]


def get_k_class(k: int, mi: int, mx: int):
    """

    :param k: k 类
    :param mi: 最小值
    :param mx: 最大值
    :return:
    """
    lst = []
    i = 0
    while i < k:
        ri = random.randint(mi, mx)
        if ri not in lst:
            lst.append(ri)
            i += 1
    return lst


def k_means(mat: np.ndarray, k: int, iters: int):
    """
    实现k-means算法
    :param iters: 迭代次数
    :param mat: 聚类的数值
    :param k: 聚类的个数
    :return:
    """
    k_ = get_k_class(k, 0, mat.shape[0])
    dots = ndarray_to_list(mat)
    clusters = [dots[i] for i in k_]
    i = 0
    while i < iters:
        print("iter", i)
        update_k_means(dots, clusters)
        i += 1
    return clusters
