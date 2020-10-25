import numpy as np


class Matrix:

    @staticmethod
    def svd(ar: np.ndarray) -> tuple:
        """
        进行矩阵奇异值分解
        :param ar: 需要进行分解的矩阵
        :return: U, Lambda, V
        """
        return np.linalg.svd(ar)

    @staticmethod
    def inv(ar: np.ndarray) -> tuple:
        """
        返回矩阵的逆
        :param ar: 需要计算逆的矩阵
        :return: 矩阵的逆
        """
        return np.linalg.inv(ar)

    @staticmethod
    def cov(ar: np.ndarray) -> np.ndarray:
        """
        计算一个矩阵的协方差，默认矩阵的行向量是一个n维随机变量
        :param ar: 需要计算协方差的矩阵
        :return: 矩阵的协方差
        """
        # return np.cov(np.transpose(ar))
        a = ar.mean(0)
        a = ar - np.ones((ar.shape[0], 1)).dot([a])
        return a.T.dot(a)

