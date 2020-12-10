import cv2
import numpy as np


class ImageController:
    """
    用于处理单一图像的 controller
    """

    def __init__(self, file: str = None, matrix: np.ndarray = None, clr: str = None):
        if file is not None:
            if clr == "GRAY":
                self.__img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            else:
                self.__img = cv2.imread(file, cv2.IMREAD_COLOR)
            if isinstance(self.__img, type(None)):
                raise ValueError("当前路径 " + file + " 不是一张图片！")
        else:
            self.__img = matrix
        if clr is None:
            self.__color_space = "BGR"
        else:
            self.__color_space = clr

    def cvt(self, clr):
        try:
            func = getattr(self, f"cvt_{clr}")
            return func()
        except Exception as _:
            print(_)
            return self

    def cvt_HLS(self):
        """
        将图片转换成HLS空间
        :return: self
        """
        if self.__color_space == "HLS":
            return self
        if self.__color_space in ["GRAY", "HSV", "LAB"]:
            self.cvt_BGR()
            self.__img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2HLS)
            self.__color_space = "HLS"
            return self
        try:
            num = getattr(cv2, f"COLOR_{self.__color_space}2HLS")
            self.__img = cv2.cvtColor(self.__img, num)
            self.__color_space = "HLS"
        except Exception as _:
            print(_)
        return self

    def cvt_HSV(self):
        """
        将图片转换到HSV空间
        :return: self
        """
        if self.__color_space == "HSV":
            return self
        if self.__color_space in ["GRAY", "HLS", "LAB"]:
            self.cvt_BGR()
            self.__img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2HSV)
            self.__color_space = "HSV"
            return self
        try:
            num = getattr(cv2, f"COLOR_{self.__color_space}2HSV")
            self.__img = cv2.cvtColor(self.__img, num)
            self.__color_space = "HSV"
        except Exception as _:
            print(_)
        return self

    def cvt_GRAY(self):
        """
        将图像转换到灰度空间
        :return: self
        """
        if self.__color_space == "GRAY":
            return self
        if self.__color_space in ["HSV", "HLS", "LAB"]:
            self.cvt_BGR()
            self.__img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)
            self.__color_space = "GRAY"
            return self
        try:
            num = getattr(cv2, f"COLOR_{self.__color_space}2GRAY")
            self.__img = cv2.cvtColor(self.__img, num)
            self.__color_space = "GRAY"
        except Exception as _:
            print(_)
        return self

    def cvt_LAB(self):
        """
        将图像转换成 lab 颜色空间
        :return: self
        """
        if self.__color_space == "LAB":
            return self
        if self.__color_space in ["HSV", "HLS", "GRAY"]:
            self.cvt_BGR()
            self.__img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2LAB)
            self.__color_space = "LAB"
            return self
        try:
            num = getattr(cv2, f"COLOR_{self.__color_space}2LAB")
            self.__img = cv2.cvtColor(self.__img, num)
            self.__color_space = "LAB"
        except Exception as _:
            print(_)
        return self

    def cvt_RGB(self):
        """
        将图像转换成 rgb 颜色空间
        :return: self
        """
        if self.__color_space == "RGB":
            return self
        try:
            num = getattr(cv2, f"COLOR_{self.__color_space}2RGB")
            self.__img = cv2.cvtColor(self.__img, num)
            self.__color_space = "RGB"
        except Exception as _:
            print(_)
        return self

    def cvt_BGR(self):
        """
        将图像转换成 bgr 颜色空间
        :return: self
        """
        if self.__color_space == "BGR":
            return self
        try:
            num = getattr(cv2, f"COLOR_{self.__color_space}2BGR")
            self.__img = cv2.cvtColor(self.__img, num)
            self.__color_space = "BGR"
        except Exception as _:
            print(_)
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
        img = self.__img
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        out = cv2.resize(img, (width, height))
        if save:
            cv2.imwrite(out_path, out)
        ic = ImageController()
        ic.__img = out
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
        img = self.__img
        out = cv2.resize(img, (width, height))
        if save:
            cv2.imwrite(out_path, out)
        ic = ImageController()
        ic.__img = out
        return ic

    def as_vector(self) -> np.ndarray:
        """
        将图片变为向量
        :return:
        """
        if self.__img is None:
            return None
        shape = self.__img.shape
        return self.__img.reshape((shape[0] * shape[1], shape[2]))

    @property
    def ndarray(self):
        return self.__img

    @property
    def clr(self):
        return self.__color_space

    def set_img(self, img: np.ndarray):
        self.__img = img

    def as_float(self):
        """
        将数组变为float形式
        :return: None
        """
        if self.__img is not None:
            self.__img = self.__img.astype(np.float32)
        return self

    def as_unit(self):
        """
        讲数组变成uint形式
        :return: None
        """
        if self.__img is not None:
            self.__img[self.__img < 0] = 0
            self.__img[self.__img > 255] = 255 * 2 - self.__img[self.__img > 255]
            self.__img = self.__img.astype(np.uint8)
        return self

    def copy(self):
        """
        返回一个自身的复制
        :return:
        """
        return ImageController(matrix=self.__img.copy(), clr=self.__color_space)
