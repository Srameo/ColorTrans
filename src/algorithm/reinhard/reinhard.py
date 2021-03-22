import cv2

from src.image_control.core.control import ImageController
from src.common_utils.core.path_utils import INPUT_PATH, OUTPUT_PATH
from src.common_utils.core.path_utils import get_root_path, path_join
import numpy as np


def mean_LAB(img: ImageController) -> tuple:
    """
    返回一个元组保存LAB的平均值
    :param img: 用于计算的图像
    :return: (ml, ma, mb)
    """
    if img.clr != "LAB":
        img.cvt_LAB()
    ml = np.mean(img.ndarray[..., 0])
    ma = np.mean(img.ndarray[..., 1])
    mb = np.mean(img.ndarray[..., 2])
    return ml, ma, mb


def std_LAB(img: ImageController) -> tuple:
    """
    返回一个元组保存LAB的标准差
    :param img: 用于计算的图像
    :return: (nl, na, nb)
    """
    if img.clr != "LAB":
        img.cvt_LAB()
    nl = np.std(img.ndarray[..., 0])
    na = np.std(img.ndarray[..., 1])
    nb = np.std(img.ndarray[..., 2])
    return nl, na, nb


def reinhard(src: ImageController, ref: ImageController) -> ImageController:
    # 将图片变为LAB
    src.cvt_LAB()
    ref.cvt_LAB()

    # 计算标准差和平均值
    src_m = mean_LAB(src)
    ref_m = mean_LAB(ref)
    src_std = std_LAB(src)
    ref_std = std_LAB(ref)

    img = np.zeros(src.ndarray.shape)

    # src 设置为 float 格式
    src.as_float()

    # 减去平均值
    img[..., 0] = src.ndarray[..., 0] - src_m[0]
    img[..., 1] = src.ndarray[..., 1] - src_m[1]
    img[..., 2] = src.ndarray[..., 2] - src_m[2]

    # 按照标准差缩放
    img[..., 0] *= ref_std[0] / src_std[0]
    img[..., 1] *= ref_std[1] / src_std[1]
    img[..., 2] *= ref_std[2] / src_std[2]

    img[np.isnan(img)] = 0

    # 加上目标图像均值
    img[..., 0] += ref_m[0]
    img[..., 1] += ref_m[1]
    img[..., 2] += ref_m[2]

    # 保存图像
    src.as_unit()
    res = ImageController(matrix=img, clr="LAB")
    return res.as_unit().cvt_BGR()


if __name__ == "__main__":
    # 获取图像
    SRC_IMG = path_join("test", "scene4", "10pm.jpg")
    REF_IMG = path_join("test", "scene4", "3pm.jpg")
    OUTPUT_IMG = path_join("test", "scene4", "3pm_10pm_reinhard.jpg")
    root_path = get_root_path()
    src_ic = ImageController(file=path_join(root_path, INPUT_PATH, SRC_IMG))
    ref_ic = ImageController(file=path_join(root_path, INPUT_PATH, REF_IMG))

    # reinhard
    cv2.imwrite(path_join(root_path, OUTPUT_PATH, OUTPUT_IMG), reinhard(src_ic, ref_ic).ndarray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
