import src.common_utils.core.image_utils as iu
import src.common_utils.core.path_utils as pu
from src.image_control.core.control import ImageController

if __name__ == "__main__":
    root_path = pu.get_root_path()

    # Test Resize
    input_filename = pu.path_join("reinhard", "flowers", "red_flower_275_183.jpeg")
    input_path = pu.path_join(root_path, pu.INPUT_PATH, input_filename)
    ic = ImageController(input_path)
    iu.print_imgs(ic.copy(),
                  ic.cvt_LAB().copy(),
                  ic.cvt_HSV().copy(),
                  ic.cvt_HLS().copy(),
                  ic.cvt_BGR().copy(),
                  ic.cvt_RGB().copy(),
                  ic.cvt_GRAY().copy())
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
