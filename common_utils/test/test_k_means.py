from common_utils.core.k_means_utils import get_k_class, ndarray_to_list, k_means
from common_utils.core.path_utils import get_root_path, path_join, INPUT_PATH
from image_control.core.control import ImageController
import numpy as np

if __name__ == "__main__":
    root_path = get_root_path()
    SRC_IMG = path_join("k_means", "flowers", "white_flower.jpg")
    ic = ImageController(path_join(root_path, INPUT_PATH, SRC_IMG))
    print(k_means(ic.reshape(), 10, 100))
