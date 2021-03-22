from src.image_control.core.img_translation import gamma_fix
import cv2
from src.common_utils.core.path_utils import INPUT_PATH, OUTPUT_PATH, get_root_path, path_join

if __name__ == '__main__':
    root_path = get_root_path()
    img = cv2.imread()
