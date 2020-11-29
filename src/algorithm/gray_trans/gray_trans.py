from src.common_utils.core import path_utils as pu
from src.common_utils.core import image_utils as iu
from src.image_control.core.control import ImageController
# from src.algorithm.reinhard.reinhard import reinhard
# from src.math_utils.core.k_means import KMeansUtil
import numpy as np
import threading

# import cv2

SRC_IMG = "gray_trans/src_img.png"
REF_IMG = "gray_trans/ref_img.png"

SWATCHES_NUM = 200
WINDOW_SIZE = 5
w1, w2 = 0.5, 0.5


class UpdateThread(threading.Thread):
    res_img = None
    reg_ref_img = None
    ref_samples = ((0, 0), 0)

    def __init__(self, low: int, high: int, tid: int):
        threading.Thread.__init__(self)
        self.low = low
        self.high = high
        self.tid = tid

    def run(self) -> None:
        UpdateThread.update_rows(self.low, self.high, self.tid)

    @classmethod
    def update_rows(cls, low: int, high: int, tid: int):
        ref_sample_loc, ref_sample_attr = cls.ref_samples
        ref_sample_x, ref_sample_y = ref_sample_loc
        h, w, c = cls.res_img.img.shape
        i, j = low, 0
        while i < high:
            print("{:.4f}% in thread {}!".format((i - low) / (high - low), tid))
            while j < w:
                # 对于每个点寻找最优点
                min_e = np.inf
                min_index = 0
                for index, attr in enumerate(ref_sample_attr):
                    e = E(attr,
                          sample_attr(cls.res_img, (i, j)))
                    if e < min_e:
                        min_e = e
                        min_index = index
                # 将alpha与beta分量赋值
                cls.res_img.img[i, j, 1] = cls.reg_ref_img.img[ref_sample_x[min_index], ref_sample_y[min_index], 1]
                cls.res_img.img[i, j, 2] = cls.reg_ref_img.img[ref_sample_x[min_index], ref_sample_y[min_index], 2]
                j += 1
            j = 0
            i += 1


def E(attr1: tuple, attr2: tuple, w1: float = w1, w2: float = w2) -> float:
    """
    计算误差
    """
    l1, std1 = attr1
    l2, std2 = attr2
    return abs(l1 - l2) * w1 + abs(std1 - std2) * w2


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
    src_img.cvt_LAB().as_float()
    ref_img.cvt_LAB().as_float()

    # 2. 将图片的亮度分布调整一致
    reg_ref_img_mat = ref_img.img.astype(np.float).copy()

    ref_mean_l = np.mean(ref_img.img[..., 0])
    src_mean_l = np.mean(src_img.img[..., 0])
    ref_std_l = np.std(ref_img.img[..., 0])
    src_std_l = np.std(src_img.img[..., 0])

    reg_ref_img_mat[..., 0] -= ref_mean_l
    reg_ref_img_mat[..., 0] *= src_std_l / ref_std_l
    reg_ref_img_mat[..., 0] += src_mean_l

    reg_ref_img = ImageController(matrix=reg_ref_img_mat, clr="LAB")

    # 3. 随机取样本
    ref_sample_x, ref_sample_y = random_swatches(reg_ref_img)

    # 4. 计算样本属性
    length = len(ref_sample_x)
    ref_sample_attr = []
    for i in range(length):
        ref_sample_attr.append(sample_attr(reg_ref_img,
                                           (ref_sample_x[i], ref_sample_y[i])))

    # 5. 寻找最优解，给颜色赋值
    h_src, w_src, c_src = src_img.img.shape
    res_img = src_img.copy()
    threads_num = 20
    low = 0
    step = int(h_src / threads_num)
    threads = []
    UpdateThread.res_img = res_img
    UpdateThread.reg_ref_img = reg_ref_img
    UpdateThread.ref_samples = (ref_sample_x, ref_sample_y), ref_sample_attr
    for i in range(threads_num - 1):
        thread = UpdateThread(low, low + step, i + 1)
        thread.start()
        threads.append(thread)
        low += step
    thread = UpdateThread(low, h_src, threads_num)
    thread.start()
    threads.append(thread)

    for t in threads:
        t.join()

    src_img.as_unit().cvt_GRAY()
    ref_img.as_unit().cvt_BGR()
    return res_img.as_unit().cvt_BGR()


if __name__ == '__main__':
    root_path = pu.get_root_path()
    src_img = ImageController(pu.path_join(root_path, pu.INPUT_PATH, SRC_IMG), clr="GRAY")
    ref_img = ImageController(pu.path_join(root_path, pu.INPUT_PATH, REF_IMG))

    res_img = gray_trans(src_img, ref_img)

    iu.print_imgs(src_img.img, ref_img.img, res_img.img)
