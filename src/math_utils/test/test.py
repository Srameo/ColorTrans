import cv2

from src.image_control.core.control import ImageController
from src.math_utils.core.matrix import Matrix
from src.common_utils.core.path_utils import INPUT_PATH, OUTPUT_PATH, get_root_path, path_join

if __name__ == '__main__':
    # 获取图像
    SRC_IMG = path_join("reinhard", "sea", "day.png")
    REF_IMG = path_join("reinhard", "sea", "night.png")
    root_path = get_root_path()
    src_ic = ImageController(file=path_join(root_path, INPUT_PATH, SRC_IMG))
    ref_ic = ImageController(file=path_join(root_path, INPUT_PATH, REF_IMG))
    a = Matrix.conv2(src_ic.cvt_GRAY(), Matrix.SOBEL_KERNEL_X)
    cv2.imshow("conv2", a)
    cv2.waitKey()
    cv2.destroyAllWindows()
