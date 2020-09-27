from PIL import Image


class ImageController:
    """
    用于处理单一图像的 controller
    """

    img = None

    def __init__(self, file: str = None):
        if file is not None:
            self.img = Image.open(file)

    def resize(self, out_path: str, *args: int) -> None:
        """
        resize 图片
        :param out_path: 输出路径
        :param args: 参数
        :return: None
        """
        if len(args) == 1:
            # 按照比例缩放
            self.resize_image_scale(out_path, args[0])
            pass
        elif len(args) == 2:
            # 按照 width 和 height 调整大小
            self.resize_image_size(out_path, args[0], args[1])
            pass
        else:
            pass

    def resize_image_scale(self, out_path: str, scale: float = 1) -> None:
        """
        改变图像大小(按照比例)
        :param out_path: 输出路径
        :param scale: 缩放比例
        :return: None
        """
        img = self.img
        width = int(img.size[0] * scale)
        height = int(img.size[1] * scale)
        tp = img.format
        out = img.resize((width, height), Image.ANTIALIAS)
        out.save(out_path, tp)

    def resize_image_size(self, out_path: str, width: int, height: int) -> None:
        """
        改变图像大小(按照给定图像大小)
        :param height: 新图像的高度
        :param width: 新图像的宽度
        :param out_path: 输出路径
        :return: None
        """
        img = self.img
        tp = img.format
        out = img.resize((width, height), Image.ANTIALIAS)
        out.save(out_path, tp)
