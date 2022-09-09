import string
import random
from io import BytesIO

import numpy as np
import cv2
import PIL
import qrcode
import barcode
from barcode.writer import ImageWriter

import paddleclas


def get_region(img, percent_x=None, percent_y=None, area_ratio=None, aspect_ratio=None):
    """random a region from the image

    Args:
        img (numpy.ndarray): the image
        percent_x (float, optional): the center location of region. Defaults to None.
        percent_y (float, optional): the center location of region. Defaults to None.
        area_ratio (float, optional): the area ratio of region to image. Defaults to None.
        aspect_ratio (float, optional): the ratio of width to height. Defaults to None.

    Returns:
        tuple: the location of region
    """
    height, width, channels = img.shape

    if percent_x:
        coordinate_x = int(width * percent_x)
    else:
        coordinate_x = int(width * random.random())
    if percent_y:
        coordinate_y = int(height * percent_y)
    else:
        coordinate_y = int(height * random.random())

    max_w = 2 * min(width - coordinate_x, coordinate_x)
    min_w = int(width * 0.1)
    min_w = min_w if min_w < max_w else max_w
    w = random.randint(min_w, max_w) // 2 * 2

    if area_ratio:
        area = int(height * width * area_ratio)
        h = int(area / w)
    elif aspect_ratio:
        h = int(w / aspect_ratio)
    else:
        max_h = 2 * min(height - coordinate_y, coordinate_y)
        min_h = int(height * 0.1)
        min_h = min_h if min_h < max_h else max_h
        h = random.randint(min_h, max_h) // 2 * 2

    w = w if (coordinate_x - w / 2) >= 0 else coordinate_x * 2
    h = h if (coordinate_y - h / 2) >= 0 else coordinate_y * 2
    x = int(coordinate_x - w / 2)
    y = int(coordinate_y - h / 2)

    return (x, y, w, h)


def replace_color(img, src_clr, dst_clr):
    """replace the src_clr color of img by dst_clr color

    Args:
        img (numpy.ndarray): the original image
        src_clr (tuple): the original color
        dst_clr (tuple): the destination color

    Returns:
        numpy.ndarray: the modified image
    """
    img_arr = np.asarray(img, dtype=np.double)

    r_img = img_arr[:,:,0].copy()
    g_img = img_arr[:,:,1].copy()
    b_img = img_arr[:,:,2].copy()

    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2] #编码

    r_img[img == src_color] = dst_clr[0]
    g_img[img == src_color] = dst_clr[1]
    b_img[img == src_color] = dst_clr[2]

    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1,2,0)

    return dst_img


def paste(back_img, fore_img, region, transparent=True):
    """paste the fore_img to back_img

    Args:
        back_img (numpy.ndarray): image as background
        fore_img (numpy.ndarray): image as foreground
        region (list): indicate the location of the paste
        transparent (bool, optional): whether the background of fore_img can be transparent. Defaults to True.

    Returns:
        numpy.ndarray: the synthesized image
    """
    x, y, w, h = region
    colors = list(PIL.ImageColor.colormap.keys())
    if transparent:
        colors += ["transparent"]

    bg_c, fg_c = random.choices(colors, k=2)
    if fg_c == "transparent":
        fg_c, bg_c = bg_c, fg_c

    fg_rgb = PIL.ImageColor.getrgb(PIL.ImageColor.colormap[fg_c])
    fore_img = replace_color(fore_img, (0, 0, 0), fg_rgb)

    if bg_c == "transparent":
        roi = back_img[y:y+h, x:x+w, :]
        mask = fore_img[:, :, 0]
        mask_inv = cv2.bitwise_not(mask)
        roi = cv2.bitwise_and(roi, roi, mask=mask)
        fore_img = cv2.bitwise_and(fore_img, fore_img, mask=mask_inv)
        dst = cv2.add(roi, fore_img)
    else:
        bg_rgb = PIL.ImageColor.getrgb(PIL.ImageColor.colormap[bg_c])
        fore_img = replace_color(fore_img, (255, 255, 255), bg_rgb)
        dst = fore_img

    dst = cv2.resize(dst, (w, h))
    back_img[y:y+h, x:x+w, :] = dst
    return back_img


def median_blur(img):
    ksize = random.randint(3, 10) // 2 * 2 + 1
    img = cv2.medianBlur(img, ksize)
    return img


def gaussian_blur(img):
    ksize = [random.randint(3, 10) // 2 * 2 + 1, random.randint(3, 10) // 2 * 2 + 1]
    img = cv2.GaussianBlur(img, ksize, 0)
    return img


def avg_blur(img):
    ksize = (random.randint(5, 15), random.randint(5, 15))
    img = cv2.blur(img, ksize)
    return img


def bilateral_blur(img):
    ksize = random.randint(3, 10)
    sigmaColor = random.randint(8, 20)
    sigmaSpace = random.randint(8, 20)
    img = cv2.bilateralFilter(img, ksize, sigmaColor, sigmaSpace)
    return img


class DoBlur(object):
    """blur the part of image

    Args:
        img (numpy.ndarray): the original image

    Returns:
        numpy.ndarray: modified image
    """
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, **kwargs):
        img = kwargs["img"]
        if random.random() > self.ratio:
            label = 0
        else:
            method = random.choice(["avg", "median", "gaussian", "bilateral"])
            if method == "median":
                img = median_blur(img)
            elif method == "gaussian":
                img = gaussian_blur(img)
            elif method == "bilateral":
                img = bilateral_blur(img)
            else:
                img = avg_blur(img)
            label = 1
        return {**kwargs, "img": img, "lable": label}


class DoMosaic(object):
    """add the mosaic to image

    Args:
        img (numpy.ndarray): the original image

    Returns:
        numpy.ndarray: modified image
    """
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, **kwargs):
        img = kwargs["img"]
        if random.random() > self.ratio:
            label = 0
        else:
            region = get_region(img)
            x, y, w, h = region

            method = random.choices(["mosaic"], weights=[1])
            #method = random.choices(["mosaic", "avg", "median", "gaussian", "bilateral"], weights=[4, 1, 1, 1, 1])
            if method == "avg":
                img[x:x+w, y:y+h, :] = avg_blur(img[x:x+w, y:y+h, :])
            elif method == "median":
                img[x:x+w, y:y+h, :] = median_blur(img[x:x+w, y:y+h, :])
            elif method == "gaussian":
                img[x:x+w, y:y+h, :] = gaussian_blur(img[x:x+w, y:y+h, :])
            elif method == "bilateral":
                img[x:x+w, y:y+h, :] = bilateral_blur(img[x:x+w, y:y+h, :])
            else:
                img[x:x+w, y:y+h, :] = self.mosaic(img, region)[x:x+w, y:y+h, :]
            label = 1
        return {**kwargs, "img": img, "label": label}

    def mosaic(self, img, region):
        max_neighbor = int(max(min(img.shape[0], img.shape[1]) / 40, 5))
        min_neighbor = 5
        neighbor = random.randint(min_neighbor, max_neighbor)
        x, y, w, h = region
        for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neightbour 防止溢出
            for j in range(0, w - neighbor, neighbor):
                rect = [j + x, i + y, neighbor, neighbor]
                color = img[i + y][j + x].tolist()  # 关键点1 tolist
                left_up = (rect[0], rect[1])
                right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
                cv2.rectangle(img, left_up, right_down, color, -1)
        return img


def DoNoisy(img):
    """add random noisy to image

    Args:
        img (numpy.ndarray): the original image

    Returns:
        numpy.ndarray: modified image
    """
    region = get_region(img)
    x, y, w, h = region
    method = random.choice(["gauss", "salt", "poisson", "speckle"])
    # method = random.choice(["salt"])

    if method == "gauss":
        #设置高斯分布的均值和方差
        mean = 0
        #设置高斯分布的标准差
        sigma = 25
        #根据均值和标准差生成符合高斯分布的噪声
        gauss = np.random.normal(mean, sigma,(h, w, 3))
        #给图片添加高斯噪声
        img[y:y+h, x:x+w, :] = img[y:y+h, x:x+w, :] + gauss
        #设置图片添加高斯噪声之后的像素值的范围
        img = np.clip(img, a_min=0, a_max=255)
    elif method == "salt":
        #设置添加椒盐噪声的数目比例
        s_vs_p = 0.5
        #设置添加噪声图像像素的数目
        amount = 0.04
        #添加salt噪声
        num_salt = np.ceil(amount * img[y:y+h, x:x+w, :].size * s_vs_p)
        #设置添加噪声的坐标位置
        coords = [np.random.randint(0,i - 1, int(num_salt)) for i in img[y:y+h, x:x+w, :].shape]
        img[y:y+h, x:x+w, :][coords] = 255
        #添加pepper噪声
        num_pepper = np.ceil(amount * img[y:y+h, x:x+w, :].size * (1. - s_vs_p))
        #设置添加噪声的坐标位置
        coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in img[y:y+h, x:x+w, :].shape]
        img[y:y+h, x:x+w, :][coords] = 0
    elif method == "poisson":
        #计算图像像素的分布范围
        vals = len(np.unique(img[y:y+h, x:x+w, :]))
        vals = 2 ** np.ceil(np.log2(vals))
        #给图片添加泊松噪声
        img[y:y+h, x:x+w, :] = np.random.poisson(img[y:y+h, x:x+w, :] * vals) / float(vals)
    else:
        #随机生成一个服从分布的噪声
        gauss = np.random.randn(h, w, 3)
        #给图片添加speckle噪声
        img[y:y+h, x:x+w, :] = img[y:y+h, x:x+w, :] + img[y:y+h, x:x+w, :] * gauss
        #归一化图像的像素值
        img = np.clip(img, a_min=0, a_max=255)

    return img


def DoBarcode(img):
    """random generate barcode and paste on img

    Args:
        img (numpy.ndarray): original image

    Returns:
        numpy.ndarray: modified image
    """
    region = get_region(img)

    # bar_formats = {u"code39": None, u"code128": None, u"PZN": None ,u"ean13": 12, u"ean8": 12, u"jan": None, u"isbn13": None, u"isbn10": None, u"issn": None, u"upca": None, u"ean14": 12, u"gs1128": None}
    bar_formats = {u"code39": None, u"code128": None, u"ean13": 12, u"ean8": 12, u"isbn13": None, u"isbn10": None, u"issn": None, u"upca": None, u"ean14": 12, u"Gs1_128": None}
    bar_format = random.choice(list(bar_formats.keys()))

    if not bar_formats[bar_format]:
        max_length_limit = 25
        length_limit = random.randint(5, max_length_limit)
    else:
        length_limit = bar_formats[bar_format] - 3
    data = "".join(random.choices(string.digits, k=length_limit))

    bar = barcode.get(bar_format, f"978{data}",writer=ImageWriter())
    bar_img = bar.render(writer_options={"format": "PNG"}) #渲染生成图像对象

    bar_img = np.array(bar_img)

    img = paste(img, bar_img, region)
    return img


def DoQrcode(img):
    """random generate QR code and paste on img

    Args:
        img (numpy.ndarray): original image

    Returns:
        numpy.ndarray: modified image
    """
    region = get_region(img)

    qr = qrcode.QRCode()
    qr.add_data(f"{random.random()}")
    qr_img = np.array(qr.make_image().convert('RGB'))

    img = paste(img, qr_img, region)
    return img


class DoRotate(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        img = kwargs["img"]
        label = random.choice([0, 1, 2, 3])
        if label:
            img = np.rot90(img, label)
        return {**kwargs, "img": img, "label": label}


class DoCorrecte(object):
    def __init__(self):
        self.model = paddleclas.PaddleClas(model_name="image_orientation")
    
    def __call__(self, **kwargs):
        img = kwargs["img"]
        result = self.model.predict(input_data=img)
        class_ids = next(result)[0]["class_ids"][0]
        img = np.rot90(img, class_ids * -1)
        return {**kwargs, "img": img}


class DoStretch(object):
    def __init__(self, f=2):
        self.f = float(f)
        self.f_ = 1 / self.f
    
    def __call__(self, **kwargs):
        img = kwargs["img"]
        height, width, channels = img.shape
        ratio = np.float32(height / width)
        new_r = random.uniform(self.f_ * ratio, self.f * ratio)
        if new_r > ratio:
            new_height = int(new_r * height)
            new_width = width
        else:
            new_height = height
            new_width = int(height / new_r)
        
        img = cv2.resize(img, (new_width, new_height))
        return {**kwargs, "img": img, "label": ratio}


def main():
    import os
    func = DoStretch()

    list_path = "/paddle/data/clas/ILSVRC2012_val/val_list.txt"
    dir_path = "/paddle/data/clas/ILSVRC2012_val/"
    with open(list_path) as f:
        lines = f.readlines()

    new_dir = "/paddle/data/Strech/"
    new_list = []
    new_list_path = "/paddle/data/Strech/val_list.txt"
    for line in lines:
        img_path = line.strip().split()[0]
        img = cv2.imread(os.path.join(dir_path, img_path))
        data = func(**{"img": img})
        img = data["img"]
        label = data["label"]
        new_list.append(f"{img_path} {label}")
        img_path = os.path.join(new_dir, img_path)
        parent_dir = os.path.dirname(img_path)
        os.makedirs(parent_dir, exist_ok=True)
        cv2.imwrite(img_path, img)

    with open(new_list_path, "w+") as f:
        f.write("\n".join(new_list))


if __name__ == "__main__":
    main()
