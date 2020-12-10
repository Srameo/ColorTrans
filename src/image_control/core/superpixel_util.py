import cv2


class SuperPixelUtil:

    def __init__(self, parent):
        self.__parent = parent

    def slic(self, region_size: int = 20, ruler: float = 20.0, iter: int = 10):
        """
        使用 SLIC 进行超像素分割
        :param region_size: 平均超像素大小，默认20
        :param ruler: 超像素平滑度，默认20
        :param iter: 迭代次数，默认10
        :return: slic
        """
        slic = cv2.ximgproc.createSuperpixelSLIC(self.__parent.ndarray,
                                                 region_size=region_size,
                                                 ruler=ruler)
        slic.iterate(iter)
        return slic

    def seeds(self, num_superpixels, num_levels, histogram_bins=5, double_step=True, iter=10):
        """
        使用 SEEDS 进行超像素分割, 多用于指定个数的超像素区域
        :param num_superpixels: 期望的超像素个数
        :param num_levels: 块级别数，值越高，分段越准确，形状越平滑，但需要更多的内存和CPU时间
        :param histogram_bins: 直方图bins数，默认5
        :param double_step: 如果为true，则每个块级别重复两次以提高准确性默认false。
        :param iter: 迭代次数
        :return: seeds
        """
        img = self.__parent.ndarray
        seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1],
                                                   img.shape[0],
                                                   img.shape[2],
                                                   num_superpixels,
                                                   num_levels,
                                                   3, histogram_bins, double_step)
        seeds.iterate(img, iter)
        return seeds

    def lsc(self, region_size=20, ratio=0.075, iter=10):
        """
        使用 lsc 方法进行超像素分割
        :param iter: 迭代次数，默认10
        :param region_size: 超像素大小，默认20
        :param ratio: 超像素紧凑度因子，默认0.075
        :return: lsc
        """
        lsc = cv2.ximgproc.createSuperpixelLSC(self.__parent.ndarray,
                                               region_size,
                                               ratio)
        lsc.iterate(iter)
        return lsc