from src.image_control.core.control import ImageController
import src.math_utils.core.matrix
import numpy as np


def mean_RGB(img: ImageController) -> tuple:
    """
    返回一个元组保存RGB的平均值
    :param img: 用于计算的图像
    :return: (ml, ma, mb)
    """
    if img.color_space != "RGB":
        img.cvt_RGB()
    mr = np.mean(img.img[..., 0])
    mg = np.mean(img.img[..., 1])
    mb = np.mean(img.img[..., 2])
    return mr, mg, mb


