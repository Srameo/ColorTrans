from src.common_utils import k_means
from src.common_utils import get_root_path, path_join, INPUT_PATH, OUTPUT_PATH
from src.image_control import ImageController
import numpy as np
import cv2

if __name__ == "__main__":
    root_path = get_root_path()
    SRC_IMG = path_join("k_means", "flowers", "white_flower.jpg")
    # SRC_IMG = path_join("k_means", "flowers", "red_flower.jpg")
    ic = ImageController(path_join(root_path, INPUT_PATH, SRC_IMG))
    # 聚类个数
    k = 10
    a, b = k_means(ic.reshape(), k, 100)
    result = np.zeros((1, k, 3))
    for index, cluster in enumerate(b):
        c = np.uint8(cluster["color"])
        result[0, index, 0] = c[0]
        result[0, index, 1] = c[1]
        result[0, index, 2] = c[2]
    print(result)
    cv2.imwrite(path_join(root_path, OUTPUT_PATH, path_join("k_means", "flowers", "white_flower.jpg")), result)
    # cv2.imwrite(path_join(root_path, OUTPUT_PATH, path_join("k_means", "flowers", "red_flower.jpg")), result)
