import numpy as np


class Matrix:
    SOBEL_KERNEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    SOBEL_KERNEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

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

    @staticmethod
    def conv2(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        计算一个图片和一个卷积核的二维卷积
        :param img: 输入的图片
        :param kernel: 卷积核
        :return: 结果
        """
        H, W = img.shape
        n = kernel.shape[0]
        col = np.zeros(H)
        raw = np.zeros(W + 2)
        img = np.insert(img, W, values=col, axis=1)
        img = np.insert(img, 0, values=col, axis=1)
        img = np.insert(img, H, values=raw, axis=0)
        img = np.insert(img, 0, values=raw, axis=0)
        res = np.zeros([H, W])  # 直接新建一个全零数组，省去了后边逐步填充数组的麻烦
        i, j = 0, 0
        while i < H:
            while j < W:
                temp = img[i:i + n, j:j + n]
                temp = np.multiply(temp, kernel)
                res[i][j] = temp.sum()
                j = j + 1
            i = i + 1

        return res
