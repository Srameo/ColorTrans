"""
用于解决有亮点的部分
"""
import numpy as np

from src.algorithm.gray_trans.gray_trans import UpdateThread, E
from src.common_utils.core.decorator import timer
from src.image_control.core.control import ImageController
from src.common_utils.core import path_utils as pu
from src.common_utils.core import image_utils as iu
from src.math_utils.core.kdtree import KDTreeUtil

# SRC_IMG = "gray_trans/src_img.png"
# REF_IMG = "gray_trans/ref_img.png"
SRC_IMG = "gray_trans/coast_src.png"
REF_IMG = "gray_trans/coast_ref.png"

ITER = 100
SRC_SUPERPIXEL_NUM = 1500
REF_SUPERPIXEL_NUM = 1500
THREADS_NUM = 10
w = np.array([[0.1], [0.5]])


class UpdateSuperPixelThread(UpdateThread):
    res_img = None
    ref_img = None
    src_samples = None
    src_superpixel_label = None
    ref_samples = None
    ref_superpixel_label = None
    kdtree = None

    def run(self) -> None:
        print("thread {} start!".format(self.tid))
        UpdateSuperPixelThread.update_rows(self.low, self.high, self.tid)
        print("thread {} end!".format(self.tid))

    @classmethod
    def update_rows(cls, low: int, high: int, tid: int):
        i = low
        while i < high:
            print("{:.4f}% in thread {}!".format((i - low) * 100 / (high - low), tid))
            # 对于每个超像素寻找最优点
            min_ = cls.kdtree.query(cls.src_samples[i])
            min_index = cls.ref_superpixel_label == min_
            samples = cls.ref_img.ndarray[min_index]
            mean_alpha = np.mean(samples[:, 1])
            mean_beta = np.mean(samples[:, 2])
            src_index = cls.src_superpixel_label == i
            temp = cls.res_img.ndarray[src_index]
            temp[:, 1] = mean_alpha
            temp[:, 2] = mean_beta
            cls.res_img.ndarray[src_index] = temp
            i = i + 1


def all_sample_attrs_std(img: ImageController, label: np.ndarray, length: int):
    ans = []
    img = img.ndarray
    for i in range(length):
        index = label == i
        data = img[index]
        mean_l = np.mean(data)
        std_l = np.std(data)
        ans.append(np.array([[mean_l, std_l]]))
    return ans


@timer
def gray_trans_superpixel(src_img: ImageController, ref_img: ImageController,
                          src_pixels: int = SRC_SUPERPIXEL_NUM, ref_pixels: int = REF_SUPERPIXEL_NUM,
                          iter: int = ITER):
    """
    给灰度图像上色，使用superpixel修正图片
    :param iter: 迭代次数
    :param ref_pixels: 参考超像素个数
    :param src_pixels: 源图像超像素个数
    :param src_img: 源图像
    :param ref_img: 参考图像
    :return:
    """
    all_sample_attrs = all_sample_attrs_std
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

    # 3. 使用superpixel分割图片
    # src_seeds = src_img.superpixel.seeds(src_pixels, 15, iter=iter)
    # ref_seeds = ref_img.superpixel.seeds(ref_pixels, 15, iter=iter)
    src_seeds = src_img.superpixel.slic(10, iter=iter)
    ref_seeds = ref_img.superpixel.slic(10, iter=iter)

    # 4. 计算样本属性
    src_seeds_label = src_seeds.getLabels()
    ref_seeds_label = ref_seeds.getLabels()
    src_sample_attrs = all_sample_attrs(src_img, src_seeds_label, src_seeds.getNumberOfSuperpixels())
    ref_sample_attrs = all_sample_attrs(ref_img, ref_seeds_label, ref_seeds.getNumberOfSuperpixels())

    # 5. 随机取样本
    hi = src_seeds.getNumberOfSuperpixels()
    res_img = src_img.copy()
    threads_num = THREADS_NUM
    low = 0
    step = int(hi / threads_num)
    threads = []
    # 初始化线程公共变量
    UpdateSuperPixelThread.res_img = res_img
    UpdateSuperPixelThread.ref_img = ref_img
    UpdateSuperPixelThread.ref_samples = ref_sample_attrs
    UpdateSuperPixelThread.ref_superpixel_label = ref_seeds_label
    UpdateSuperPixelThread.src_samples = src_sample_attrs
    UpdateSuperPixelThread.src_superpixel_label = src_seeds_label
    UpdateSuperPixelThread.kdtree = KDTreeUtil(ref_sample_attrs)
    # 创建线程
    for i in range(threads_num - 1):
        thread = UpdateSuperPixelThread(low, low + step, i + 1)
        thread.start()
        threads.append(thread)
        low += step
    thread = UpdateSuperPixelThread(low, hi, threads_num)
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

    res_img = gray_trans_superpixel(src_img, ref_img)

    iu.print_imgs(src_img, ref_img, res_img)
    iu.save_img(res_img.ndarray, pu.path_join(root_path, pu.OUTPUT_PATH, f"gray_trans/superpixel/coast/res_img"
                                                                         f"_{w[0][0]}"
                                                                         f"_{w[1][0]}"
                                                                         f"_{ SRC_SUPERPIXEL_NUM }"
                                                                         f"_{ REF_SUPERPIXEL_NUM }.png"))
