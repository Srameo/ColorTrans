import cv2

from src.common_utils.core.path_utils import get_root_path, path_join, INPUT_PATH, OUTPUT_PATH
from src.image_control.core.control import ImageController
from src.math_utils.core.matrix import Matrix
import numpy as np

SRC_IMG = "rgb_trans/src_img.JPG"
REF_IMG = "rgb_trans/ref_img.JPG"


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


def rgb_trans(src_img: ImageController, ref_img: ImageController) -> ImageController:
    """
    Color Transfer in Correlated Color Space
    :param src_img: 源图像
    :param ref_img: 参考图像
    :return: 结果
    """
    # 获取初始变量
    src_img.cvt_RGB()
    ref_img.cvt_RGB()
    src_vec = src_img.as_vector()
    ref_vec = ref_img.as_vector()
    r_bar_src, g_bar_src, b_bar_src = mean_RGB(src_img)
    r_bar_ref, g_bar_ref, b_bar_ref = mean_RGB(ref_img)
    cov_src = Matrix.cov(src_vec)
    cov_ref = Matrix.cov(ref_vec)
    u_src, lambda_src, v_src = Matrix.svd(cov_src)
    u_ref, lambda_ref, v_ref = Matrix.svd(cov_ref)

    # 获取中间变量
    t_src = np.array([[1, 0, 0, -r_bar_src],
                      [0, 1, 0, -g_bar_src],
                      [0, 0, 1, -b_bar_src],
                      [0, 0, 0, 1]])
    t_ref = np.array([[1, 0, 0, r_bar_ref],
                      [0, 1, 0, g_bar_ref],
                      [0, 0, 1, b_bar_ref],
                      [0, 0, 0, 1]])
    r_src = np.vstack((np.hstack((Matrix.inv(u_src), np.zeros((3, 1)))),
                       np.hstack((np.zeros((1, 3)), [[1]]))))
    r_ref = np.vstack((np.hstack((u_ref, np.zeros((3, 1)))),
                       np.hstack((np.zeros((1, 3)), [[1]]))))
    s_src = np.diag(np.hstack((1 / np.sqrt(lambda_src), [1])))
    s_ref = np.diag(np.hstack((np.sqrt(lambda_ref), [1])))

    # 将矩阵变为齐次坐标
    src_vec_corr = np.hstack((src_vec, np.ones((ref_vec.shape[0], 1))))

    # 计算数据
    res_vec_corr = t_ref.dot(r_ref.dot(s_ref.dot(s_src.dot(r_src.dot(t_src.dot(np.transpose(src_vec_corr)))))))
    res_vec_corr = np.transpose(res_vec_corr)

    # 将结果变为齐次坐标
    # shape = res_vec_corr.shape
    # res_vec = np.zeros((shape[0], 3))
    # res_vec[::, 0] = res_vec_corr[::, 0] / res_vec_corr[::, 3]
    # res_vec[::, 1] = res_vec_corr[::, 1] / res_vec_corr[::, 3]
    # res_vec[::, 2] = res_vec_corr[::, 2] / res_vec_corr[::, 3]
    res_vec = res_vec_corr[::, 0:3]
    res_mat = res_vec.reshape(src_img.img.shape)

    res = ImageController(matrix=res_mat, clr="RGB").as_unit()
    return res.cvt_BGR()


if __name__ == '__main__':
    # 获取图像
    root_path = get_root_path()
    src_ic = ImageController(file=path_join(root_path, INPUT_PATH, SRC_IMG))
    ref_ic = ImageController(file=path_join(root_path, INPUT_PATH, REF_IMG))
    output_path = path_join(root_path, OUTPUT_PATH, "rgb_trans", )

    cv2.imshow("src", src_ic.img)
    cv2.waitKey()
    cv2.imshow("ref", ref_ic.img)
    cv2.waitKey()
    cv2.imshow("res", rgb_trans(src_ic, ref_ic).img)
    cv2.waitKey()
    cv2.destroyAllWindows()
