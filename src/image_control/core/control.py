import cv2
import numpy as np


class ImageController:
    """
    用于处理单一图像的 controller
    """

    def __init__(self, file: str = None, matrix=None, clr=None):
        if file is not None:
            self.img = cv2.imread(file, cv2.IMREAD_COLOR)
            if isinstance(self.img, type(None)):
                raise ValueError("当前路径 " + file + " 不是一张图片！")
        else:
            self.img = matrix
        if clr is None:
            self.color_space = "BGR"
        else:
            self.color_space = clr

    def cvt_LAB(self):
        """
        将图像转换成 lab 颜色空间
        :return: self
        """
        if self.color_space == "RGB":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)
            self.color_space = "LAB"
        elif self.color_space == "BGR":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
            self.color_space = "LAB"
        return self

    def cvt_RGB(self):
        """
        将图像转换成 bgr 颜色空间
        :return: self
        """
        if self.color_space == "LAB":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2RGB)
            self.color_space = "RGB"
        elif self.color_space == "BGR":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.color_space = "RGB"
        return self

    def cvt_BGR(self):
        """
        将图像转换成 bgr 颜色空间
        :return: self
        """
        if self.color_space == "LAB":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2BGR)
            self.color_space = "BGR"
        elif self.color_space == "RGB":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            self.color_space = "BGR"
        return self

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
            return self.__resize_image_scale(out_path, args[0], save)
        elif len(args) == 2:
            # 按照 width 和 height 调整大小
            return self.__resize_image_size(out_path, args[0], args[1], save)
        else:
            pass

    def __resize_image_scale(self, out_path: str, scale: float, save: bool):
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

    def __resize_image_size(self, out_path: str, width: int, height: int, save: bool):
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

    def reshape(self) -> np.ndarray:
        height = self.img.shape[0]
        width = self.img.shape[1]
        colors = self.img.shape[2]
        return self.img.reshape((height * width, colors))

    def k_means(self, k):
        pass

    def as_vector(self):
        if self.img is None:
            return None
        shape = self.img.shape
        return self.img.reshape((shape[0] * shape[1], shape[2]))

    def as_ndarray(self):
        return self.img

    def as_float(self):
        """
        将数组变为float形式
        :return: None
        """
        if self.img is not None:
            self.img = self.img.astype(np.float32)
        return self

    def as_unit(self):
        """
        讲数组变成uint形式
        :return: None
        """
        if self.img is not None:
            self.img[self.img < 0] *= 0
            self.img[self.img > 255] = 255 * 2 - self.img[self.img > 255]
            self.img = self.img.astype(np.uint8)
        return self
