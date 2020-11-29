from src.common_utils.core import path_utils as pu
from src.common_utils.core import image_utils as iu
from src.image_control.core.control import ImageController
from src.algorithm.reinhard.reinhard import reinhard
from src.math_utils.core.k_means import KMeansUtil
import numpy as np
import cv2

SRC_IMG = "gray_trans/src_img.png"
REF_IMG = "gray_trans/ref_img.png"

SWATCHES_NUM = 100
WINDOW_SIZE = 5
w1, w2 = 1, 0.5


def E(attr1: tuple, attr2: tuple, w1: float = w1, w2: float = w2) -> float:
    l1, std1 = attr1
    l2, std2 = attr2
    return abs(l1.astype(np.float) - l2.astype(np.float)) * w1 + \
           abs(std1 - std2) * w2


def sample_attr(img: ImageController, loc: tuple, wd_size: int = WINDOW_SIZE) -> tuple:
    """
    计算样本的属性
    :param img: 图像
    :param loc: 样本坐标
    :param wd_size: window size
    :return: 样本的l, window中的方差
    """
    x, y = loc
    h, w, c = img.img.shape
    l = img.img[x, y, 0]
    x_begin = x - int(wd_size / 2) if x > int(wd_size / 2) else 0
    y_begin = y - int(wd_size / 2) if y > int(wd_size / 2) else 0
    x_end = x + int(wd_size / 2) + 1 if x + int(wd_size / 2) < h else h
    y_end = y + int(wd_size / 2) + 1 if y + int(wd_size / 2) < w else w
    std = np.std(img.img[x_begin:x_end, y_begin:y_end, 0])
    return l, std


def random_swatches(img: ImageController, swa_num: int = SWATCHES_NUM):
    """
    随机去样本
    :param img: 图片
    :param swa_num: 样本个数
    :return:
    """
    h, w, c = img.img.shape
    x = np.random.randint(low=0, high=h, size=swa_num)
    y = np.random.randint(low=0, high=w, size=swa_num)
    return x, y


def gray_trans(src_img: ImageController, ref_img: ImageController) -> ImageController:
    """
    给灰度图像上色
    :param src_img: 原图像
    :param ref_img: 参考图像
    :return:
    """
    # 1. 将图片转换到LAB空间
    src_img.cvt_LAB()
    ref_img.cvt_LAB()

    # 2. 将图片的颜色分布调整一致, 此处使用 reinhard
    reg_src_img = reinhard(src_img, ref_img).cvt_LAB()

    # 3. 随机取样本
    ref_sample_x, ref_sample_y = random_swatches(ref_img)

    # 4. 计算样本属性
    length = len(ref_sample_x)
    ref_sample_attr = []
    for i in range(length):
        ref_sample_attr.append(sample_attr(ref_img,
                                           (ref_sample_x[i], ref_sample_y[i])))

    # 5. 寻找最优解，给颜色赋值
    h_src, w_src, c_src = reg_src_img.img.shape
    res_img = reg_src_img.copy()
    i, j = 0, 0
    while i < h_src:
        while j < w_src:
            # 对于每个点寻找最优点
            min_e = np.inf
            min_index = 0
            for index, attr in enumerate(ref_sample_attr):
                e = E(attr,
                      sample_attr(reg_src_img, (i, j)))
                if e < min_e:
                    min_e = e
                    min_index = index
            # 将alpha与beta分量赋值
            res_img.img[i, j, 1] = ref_img.img[ref_sample_x[min_index], ref_sample_y[min_index], 1]
            res_img.img[i, j, 2] = ref_img.img[ref_sample_x[min_index], ref_sample_y[min_index], 2]
            j += 1
        i += 1

    src_img.cvt_GRAY()
    ref_img.cvt_BGR()
    return res_img.cvt_BGR()


if __name__ == '__main__':
    root_path = pu.get_root_path()
    src_img = ImageController(pu.path_join(root_path, pu.INPUT_PATH, SRC_IMG), clr="GRAY")
    ref_img = ImageController(pu.path_join(root_path, pu.INPUT_PATH, REF_IMG))

    res_img = gray_trans(src_img, ref_img)

    iu.print_imgs(src_img.img, ref_img.img, res_img.img)
