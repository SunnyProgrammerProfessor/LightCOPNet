# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：data_augmentation.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/3/13 17:47 
"""
import sys
sys.path.append("/home/data/b532zhaoxiaohui/shuaige/ChangeDetection")
import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms
import torch

"""
CD data set with pixel-level labels；
├─pre_image 
├─post_image
├─label
└─list
"""


class MyCDDataAugmentation:
    def __init__(
            self,
            img_size,
            # with_resized=False,  # 判断是否要resize，后面通过比较img_size_dynamic来处理
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rotation=False,
            with_random_crop=False,
            with_random_blur=False,
            with_scale_random_crop=False
    ):
        self.img_size = img_size
        if self.img_size is None:  # 判断图像是否要裁剪
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        # self.with_resized = with_resized
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rotation = with_random_rotation
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.random_base = 0.5

    def transform(self, imgs, labels, to_tensor_and_norm=True):
        """
        :param imgs: [ndarray,]   1024, 1024, 3
        :param labels: [ndarray,]   1024, 1024
        :return: [ndarray,],[ndarray,]
        """
        # 将tensor或者np.ndarray转化成PIL图片
        imgs = [TF.to_pil_image(img) for img in imgs]  # imgs:(width,height) 1024, 1024
        labels = [TF.to_pil_image(label) for label in labels]
        # 动态判断图像是否要裁剪
        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]
        # 调整标签大小
        if len(labels) != 0 and self.img_size is not None:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [TF.resize(label, [self.img_size, self.img_size], interpolation=0)
                          for label in labels]

        # 水平翻转
        if self.with_random_hflip and random.random() > self.random_base:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(label) for label in labels]

        # 竖直翻转
        if self.with_random_vflip and random.random() > self.random_base:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(label) for label in labels]

        # 随机旋转
        if self.with_random_rotation and random.random() > self.random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            rotation_angle = angles[index]
            imgs = [TF.rotate(img, rotation_angle) for img in imgs]
            labels = [TF.rotate(label, rotation_angle) for label in labels]

        if self.with_random_crop and random.random() > self.random_base:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size).get_params(img=imgs[0], scale=(0.8, 1.0),
                                                                                     ratio=(1, 1))
            imgs = [TF.resized_crop(img, i, j, h, w, size=[self.img_size, self.img_size], interpolation=Image.CUBIC)
                    for img in imgs]
            labels = [
                TF.resized_crop(label, i, j, h, w, size=[self.img_size, self.img_size], interpolation=Image.NEAREST)
                for label in labels]

        if self.with_random_blur and random.random() > self.random_base:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if self.with_scale_random_crop:
            # rescale
            rescale_range = [1, 1.2]
            rescale_ratio = rescale_range[0] + random.random() * (rescale_range[1] - rescale_range[0])
            imgs = [rescale(img, rescale_ratio, order=3) for img in imgs]
            labels = [rescale(label, rescale_ratio, order=0) for label in labels]

            # crop
            imgsize = imgs[0].size
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [crop(img, box, cropsize=self.img_size, default_value=255)
                      for img in labels]

        if to_tensor_and_norm:
            imgs = [TF.to_tensor(img) for img in imgs]
            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    for img in imgs]
            labels = [torch.from_numpy(np.array(label, np.uint8)).unsqueeze(dim=0)
                      for label in labels]
        return imgs, labels


def rescale(img, scale_ratio, order):
    assert isinstance(img, Image.Image)  # 判断img类型是否正确
    assert order in [0, 3], "order must be 0 or 3"
    h, w = img.size
    scale_area = (int(np.round(h * scale_ratio)), int(np.round(w * scale_ratio)))
    if scale_area[0] == img.size[0] and scale_area[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    if order == 0:
        resample = Image.NEAREST
    return img.resize(size=(scale_area[-1], scale_area[0]), resample=resample)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    min_h = min(h, cropsize)  # 获取数据集的图片和img_size的最小值,用于剪裁
    min_w = min(w, cropsize)
    h_crop = h - min_h
    w_crop = w - min_w
    if w_crop > 0:
        cont_left = 0
        img_left = random.randrange(w_crop + 1)
    else:
        cont_left = random.randrange(-w_crop + 1)
        img_left = 0
    if h_crop > 0:
        cont_top = 0
        img_top = random.randrange(h_crop + 1)
    else:
        cont_top = random.randrange(-h_crop + 1)
        img_top = 0

    return cont_top, cont_top + min_h, cont_left, cont_left + min_w, img_top, img_top + min_h, img_left, img_left + min_w


def crop(img, box, cropsize, default_value):
    assert isinstance(img, Image.Image)
    img = np.array(img)
    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    return Image.fromarray(cont)
