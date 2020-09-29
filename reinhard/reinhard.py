from image_control.core.control import ImageController
from common_utils.core.path_utils import INPUT_PATH, OUTPUT_PATH
from common_utils.core.path_utils import get_root_path, path_join
import numpy as np
import cv2

SRC_IMG = path_join("reinhard", "sea", "day.png")
REF_IMG = path_join("reinhard", "sea", "night.png")
OUTPUT_IMG = path_join("reinhard", "sea", "result.png")


def mean_LAB(img: ImageController) -> tuple:
    """
    返回一个元组保存LAB的平均值
    :param img: 用于计算的图像
    :return: (ml, ma, mb)
    """
    if img.color_space != "LAB":
        img.cvt_LAB()
    ml = np.mean(img.img[..., 0])
    ma = np.mean(img.img[..., 1])
    mb = np.mean(img.img[..., 2])
    return ml, ma, mb


def std_LAB(img: ImageController) -> tuple:
    """
    返回一个元组保存LAB的标准差
    :param img: 用于计算的图像
    :return: (nl, na, nb)
    """
    if img.color_space != "LAB":
        img.cvt_LAB()
    nl = np.std(img.img[..., 0])
    na = np.std(img.img[..., 1])
    nb = np.std(img.img[..., 2])
    return nl, na, nb


def reinhard(src: ImageController, ref: ImageController) -> None:
    # 计算标准差和平均值
    src_m = mean_LAB(src)
    ref_m = mean_LAB(ref)
    src_std = std_LAB(src)
    ref_std = std_LAB(ref)

    # src 设置为 float 格式
    src.as_float()

    # 减去平均值
    src.img[..., 0] = src.img[..., 0] - src_m[0]
    src.img[..., 1] = src.img[..., 1] - src_m[1]
    src.img[..., 2] = src.img[..., 2] - src_m[2]

    # 按照标准差缩放
    src.img[..., 0] *= ref_std[0] / src_std[0]
    src.img[..., 1] *= ref_std[1] / src_std[1]
    src.img[..., 2] *= ref_std[2] / src_std[2]

    # 加上目标图像均值
    src.img[..., 0] = src.img[..., 0] + ref_m[0]
    src.img[..., 1] = src.img[..., 1] + ref_m[1]
    src.img[..., 2] = src.img[..., 2] + ref_m[2]

    # 保存图像
    src.as_unit()
    cv2.imwrite(path_join(root_path, OUTPUT_PATH, OUTPUT_IMG), src.cvt_RGB().img)


if __name__ == "__main__":
    # 获取图像
    root_path = get_root_path()
    src_ic = ImageController(path_join(root_path, INPUT_PATH, SRC_IMG))
    ref_ic = ImageController(path_join(root_path, INPUT_PATH, REF_IMG))

    # reinhard
    reinhard(src_ic, ref_ic)
