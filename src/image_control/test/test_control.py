import src.common_utils.core.image_utils as iu
import src.common_utils.core.path_utils as pu
from src.image_control.core.control import ImageController
import cv2

if __name__ == "__main__":
    root_path = pu.get_root_path()

    # Test Resize
    # input_filename = pu.path_join("reinhard", "flowers", "red_flower_275_183.jpeg")
    input_filename = pu.path_join("gray_trans", "src_img.png")
    input_path = pu.path_join(root_path, pu.INPUT_PATH, input_filename)
    ic = ImageController(input_path, clr="GRAY")
    slic = ic.superpixel.seeds(200, 15)
    mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
    label_slic = slic.getLabels()  # 获取超像素标签
    number_slic = slic.getNumberOfSuperpixels()  # 获取超像素数目
    mask_inv_slic = cv2.bitwise_not(mask_slic)
    img_slic = cv2.bitwise_and(ic.ndarray, ic.ndarray, mask=mask_inv_slic)  # 在原图上绘制超像素边界
    cv2.imshow("img_slic", img_slic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # iu.print_imgs(ic.copy(),
    #               ic.cvt_LAB().copy(),
    #               ic.cvt_HSV().copy(),
    #               ic.cvt_HLS().copy(),
    #               ic.cvt_BGR().copy(),
    #               ic.cvt_RGB().copy(),
    #               ic.cvt("GRAY"))
    # ic.resize(output_path, True, 64, 64)

    # Test Reshape
    # SRC_IMG = path_join("reinhard", "sea", "day.png")
    # ic = ImageController(path_join(root_path, INPUT_PATH, SRC_IMG))
    # print(ic.ndarray.shape)
    # h = ic.ndarray.shape[0]
    # w = ic.ndarray.shape[1]
    #
    # reshaped_img = ic.ndarray.reshape((h * w, 3))
    # print(reshaped_img)
    # for i in range(reshaped_img):
    #     print(reshaped_img[i])
