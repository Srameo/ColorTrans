from image_control.core.control import ImageController
from common_utils.utils import INPUT_PATH, OUTPUT_PATH
from common_utils.utils import get_root_path, path_join

if __name__ == "__main__":
    root_path = get_root_path()
    input_filename = "white_flower_500_326.jpg"
    output_filename = "white_flower.jpg"
    input_path = path_join(root_path, INPUT_PATH, input_filename)
    output_path = path_join(root_path, OUTPUT_PATH, output_filename)
    ic = ImageController(input_path)
    ic.resize(output_path, False, 0.5)
    # ic.resize(output_path, 64, 64)
