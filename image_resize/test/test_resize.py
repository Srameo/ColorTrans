from image_resize.core.resize import ImageController
from common_utils.utils import INPUT_PATH, OUTPUT_PATH
from common_utils.utils import get_root_path, path_join

if __name__ == "__main__":
    root_path = get_root_path()
    input_filename = "in_img.jpg"
    output_filename = "out_img.png"
    input_path = path_join(root_path, INPUT_PATH, input_filename)
    output_path = path_join(root_path, OUTPUT_PATH, output_filename)
    ic = ImageController(input_path)
    # ic.resize(output_path, 0.5)
    ic.resize(output_path, 64, 64)
