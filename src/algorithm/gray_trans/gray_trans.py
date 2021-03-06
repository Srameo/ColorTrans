from src.common_utils.core import path_utils as pu
from src.common_utils.core import image_utils as iu
from src.image_control.core.control import ImageController
from src.math_utils.core.matrix import Matrix
from src.common_utils.core.decorator import timer
# from src.algorithm.reinhard.reinhard import reinhard
# from src.math_utils.core.k_means import KMeansUtil
from src.math_utils.core.kdtree import KDTreeUtil
import numpy as np
import threading
# import cv2

SRC_IMG = "gray_trans/src_img.png"
REF_IMG = "gray_trans/ref_img.png"

SWATCHES_NUM = 200
WINDOW_SIZE = 5
THREADS_NUM = 1
w = np.array([[0.5], [0.5]])


class UpdateThread(threading.Thread):
    res_img = None
    reg_ref_img = None
    ref_samples = (([], []), [])
    sample_attr = None
    kdtree = None

    def __init__(self, low: int, high: int, tid: int):
        threading.Thread.__init__(self)
        self.low = low
        self.high = high
        self.tid = tid

    def run(self) -> None:
        print("thread {} start!".format(self.tid))
        UpdateThread.update_rows(self.low, self.high, self.tid)
        print("thread {} end!".format(self.tid))

    @classmethod
    def update_rows(cls, low: int, high: int, tid: int):
        ref_sample_loc, ref_sample_attr = cls.ref_samples
        ref_sample_x, ref_sample_y = ref_sample_loc
        h, w, c = cls.res_img.ndarray.shape
        i, j = low, 0
        while i < high:
            print("{:.4f}% in thread {}!".format((i - low) * 100 / (high - low), tid))
            while j < w:
                # 对于每个点寻找最优点
                # min_e = np.inf
                # min_index = 0
                # for index, attr in enumerate(ref_sample_attr):
                #     e = E(attr,
                #           cls.sample_attr(cls.res_img, (i, j)))
                #     if e < min_e:
                #         min_e = e
                #         min_index = index
                min_index = cls.kdtree.query(cls.sample_attr(cls.res_img, (i, j)))
                # 将alpha与beta分量赋值
                cls.res_img.ndarray[i, j, 1] = cls.reg_ref_img.ndarray[ref_sample_x[min_index], ref_sample_y[min_index], 1]
                cls.res_img.ndarray[i, j, 2] = cls.reg_ref_img.ndarray[ref_sample_x[min_index], ref_sample_y[min_index], 2]
                j += 1
            j = 0
            i += 1


def E(attr1: np.ndarray, attr2: np.ndarray, w: np.ndarray = w) -> float:
    """
    计算误差
    """
    return np.abs(attr1 - attr2).dot(w)


def sample_attr_std(img: ImageController, loc: tuple, wd_size: int = WINDOW_SIZE) -> np.ndarray:
    """
    计算样本的属性，计算l与std
    :param img: 图像
    :param loc: 样本坐标
    :param wd_size: window size
    :return: 样本的l, window中的方差
    """
    x, y = loc
    h, w, c = img.ndarray.shape
    l = img.ndarray[x, y, 0]
    x_begin = x - int(wd_size / 2) if x > int(wd_size / 2) else 0
    y_begin = y - int(wd_size / 2) if y > int(wd_size / 2) else 0
    x_end = x + int(wd_size / 2) + 1 if x + int(wd_size / 2) < h else h
    y_end = y + int(wd_size / 2) + 1 if y + int(wd_size / 2) < w else w
    std = np.std(img.ndarray[x_begin:x_end, y_begin:y_end, 0])
    return np.array([[l, std]])


# def sample_attr_gradient(ndarray: ImageController, loc: tuple, wd_size: int = WINDOW_SIZE) -> np.ndarray:
#     """
#     计算图片的梯度和l
#     :param ndarray: 图像
#     :param loc: 样本坐标
#     :param wd_size: window size
#     :return: l 与 gradient
#     """
#     x, y = loc
#     kernel_x = Matrix.SCHARR_KERNEL_X
#     kernel_y = Matrix.SCHARR_KERNEL_Y
#     l = ndarray.ndarray[x, y, 0]
#     mat = ndarray.ndarray
#     h, w, c = ndarray.ndarray.shape
#     x_begin = x - int(wd_size / 2) if x > int(wd_size / 2) else 0
#     y_begin = y - int(wd_size / 2) if y > int(wd_size / 2) else 0
#     x_end = x + int(wd_size / 2) + 1 if x + int(wd_size / 2) < h else h
#     y_end = y + int(wd_size / 2) + 1 if y + int(wd_size / 2) < w else w
#     gd_x = Matrix.conv2(mat[x_begin:x_end, y_begin:y_end, 0], kernel_x)
#     gd_y = Matrix.conv2(mat[x_begin:x_end, y_begin:y_end, 0], kernel_y)
#     gd = np.mean(abs(gd_x) + abs(gd_y))
#     return np.array([[l, gd]])


def random_swatches(img: ImageController, swa_num: int = SWATCHES_NUM):
    """
    随机去样本
    :param img: 图片
    :param swa_num: 样本个数
    :return:
    """
    h, w, c = img.ndarray.shape
    x = np.random.randint(low=0, high=h, size=swa_num)
    y = np.random.randint(low=0, high=w, size=swa_num)
    return x, y


@timer
def gray_trans(src_img: ImageController, ref_img: ImageController) -> ImageController:
    """
    给灰度图像上色
    :param src_img: 原图像
    :param ref_img: 参考图像
    :return:
    """
    sample_attr = sample_attr_std
    # 1. 将图片转换到LAB空间
    src_img.cvt_LAB().as_float()
    ref_img.cvt_LAB().as_float()

    # 2. 将参考图像的亮度调整至类似于原图像
    reg_ref_img_mat = ref_img.ndarray.astype(np.float).copy()

    ref_mean_l = np.mean(ref_img.ndarray[..., 0])
    src_mean_l = np.mean(src_img.ndarray[..., 0])
    ref_std_l = np.std(ref_img.ndarray[..., 0])
    src_std_l = np.std(src_img.ndarray[..., 0])

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
    h_src, w_src, c_src = src_img.ndarray.shape
    res_img = src_img.copy()
    threads_num = THREADS_NUM
    low = 0
    step = int(h_src / threads_num)
    threads = []
    # 初始化线程公共变量
    UpdateThread.res_img = res_img
    UpdateThread.reg_ref_img = reg_ref_img
    UpdateThread.ref_samples = (ref_sample_x, ref_sample_y), ref_sample_attr
    UpdateThread.sample_attr = sample_attr
    UpdateThread.kdtree = KDTreeUtil(ref_sample_attr, leaf_size=3)
    # 创建线程
    for i in range(threads_num - 1):
        thread = UpdateThread(low, low + step, i + 1)
        thread.start()
        threads.append(thread)
        low += step
    thread = UpdateThread(low, h_src, threads_num)
    thread.start()
    threads.append(thread)

    # 多线程等待结束
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

    iu.print_imgs(src_img.ndarray, ref_img.ndarray, res_img.ndarray)

    iu.save_img(res_img.ndarray, pu.path_join(root_path, pu.OUTPUT_PATH, f"gray_trans/coast/res_img_{ WINDOW_SIZE }_{ SWATCHES_NUM }.png"))

    # out1 = ImageController(pu.path_join(root_path, pu.OUTPUT_PATH, "gray_trans/trees/res_img_5_1000.png"))
    # out2 = ImageController(pu.path_join(root_path, pu.OUTPUT_PATH, "gray_trans/trees/res_img_10_1000.png"))
    #
    # diff = out1.ndarray.astype(np.int) - out2.ndarray.astype(np.int)
    # print(diff)
