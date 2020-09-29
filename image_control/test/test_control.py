from image_control.core.control import ImageController
from common_utils.core.path_utils import INPUT_PATH, OUTPUT_PATH
from common_utils.core.path_utils import get_root_path, path_join

if __name__ == "__main__":
    root_path = get_root_path()

    # Test Resize
    input_filename = path_join("reinhard", "flowers", "red_flower_275_183.jpeg")
    output_filename = path_join("reinhard", "flowers", "red_flower.jpg")
    input_path = path_join(root_path, INPUT_PATH, input_filename)
    output_path = path_join(root_path, OUTPUT_PATH, output_filename)
    ic = ImageController(input_path)
    ic.resize(output_path, True, 50 / 275)
    # ic.resize(output_path, True, 64, 64)

    # Test Reshape
    # SRC_IMG = path_join("reinhard", "sea", "day.png")
    # ic = ImageController(path_join(root_path, INPUT_PATH, SRC_IMG))
    # print(ic.img.shape)
    # h = ic.img.shape[0]
    # w = ic.img.shape[1]
    #
    # reshaped_img = ic.img.reshape((h * w, 3))
    # print(reshaped_img)
    # for i in range(reshaped_img):
    #     print(reshaped_img[i])
