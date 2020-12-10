import sys
import os

PROJECT_NAME = "ColorTrans"

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path[:cur_path.find(PROJECT_NAME + os.path.sep)+len(PROJECT_NAME + os.path.sep)]
sys.path.append(os.path.join(root_path))

from src.algorithm.gray_trans.gray_trans import gray_trans
from src.image_control.core.control import ImageController
from src.common_utils.core import path_utils as pu
from src.common_utils.core import image_utils as iu


if __name__ == '__main__':
    SRC_IMG = "gray_trans/src_img.png"
    REF_IMG = "gray_trans/ref_img.png"
    root_path = pu.get_root_path()
    src_img = ImageController(pu.path_join(root_path, pu.INPUT_PATH, SRC_IMG), clr="GRAY")
    ref_img = ImageController(pu.path_join(root_path, pu.INPUT_PATH, REF_IMG))

    res_img = gray_trans(src_img, ref_img)

    iu.print_imgs(src_img.ndarray, ref_img.ndarray, res_img.ndarray)
    # iu.save_img(res_img.ndarray, pu.path_join(root_path, pu.OUTPUT_PATH, "gray_trans/res_img.png"))
