import cv2


class ImageController:
    """
    用于处理单一图像的 controller
    """

    img = None

    def __init__(self, file: str = None):
        if file is not None:
            self.img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    def resize(self, out_path: str, save: bool, *args):
        """
        resize 图片
        :param save: 是否保存
        :param out_path: 输出路径
        :param args: 参数
        :return: ImageController
        """
        if len(args) == 1:
            # 按照比例缩放
            return self.resize_image_scale(out_path, args[0], save)
        elif len(args) == 2:
            # 按照 width 和 height 调整大小
            return self.resize_image_size(out_path, args[0], args[1], save)
        else:
            pass

    def resize_image_scale(self, out_path: str, scale: float, save: bool):
        """
        改变图像大小(按照比例)
        :param save: 是否保存
        :param out_path: 输出路径
        :param scale: 缩放比例
        :return: ImageController
        """
        img = self.img
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        out = cv2.resize(img, (width, height))
        if save:
            cv2.imwrite(out_path, out)
        ic = ImageController()
        ic.img = out
        return ic

    def resize_image_size(self, out_path: str, width: int, height: int, save: bool):
        """
        改变图像大小(按照给定图像大小)
        :param save: 是否保存
        :param height: 新图像的高度
        :param width: 新图像的宽度
        :param out_path: 输出路径
        :return: ImageController
        """
        img = self.img
        out = cv2.resize(img, (width, height))
        if save:
            cv2.imwrite(out_path, out)
        ic = ImageController()
        ic.img = out
        return ic
