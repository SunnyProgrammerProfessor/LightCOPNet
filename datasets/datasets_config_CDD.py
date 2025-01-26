# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：datasets_config.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/4/28 19:58 
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder  # 读取图片数据
from PIL import Image
import cv2
import sys
sys.path.append("/home/data/b532zhaoxiaohui/shuaige/ChangeDetection")
from datasets.utils_CDD import get_image_data
from datasets.data_augmentation import MyCDDataAugmentation
import numpy as np

"""
CD data set with pixel-level labels；
├─pre_image 
├─post_image
├─label
└─list
"""
# 这里需要自定义数据增强函数
train_augmentation = MyCDDataAugmentation(256, True, True, True, True, True, True)
val_augmentation = MyCDDataAugmentation(1024)
img_transform = {
    "train": transforms.Compose([
        train_augmentation,
    ]),
    "val": transforms.Compose([
        val_augmentation,
    ])
}


class CDDataset(Dataset):
    def __init__(self, path, form: str, img_size, transform=True, is_train=True):
        super(CDDataset, self).__init__()
        self.path = path
        self.form = form
        self.img_size = img_size
        self.is_train = is_train
        self.transform = transform
        self.imageA_path, self.imageB_path, self.imageLabel_path = get_image_data(form=form, path=path)
        assert len(self.imageA_path) == len(self.imageLabel_path)
        if self.transform and self.is_train and self.form == "train":
            self.augmentation = MyCDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_rotation=True,
                with_random_blur=True
            )
        if self.transform and self.is_train and self.form == "val":
            self.augmentation = MyCDDataAugmentation(
                img_size=self.img_size
            )
        # self.images_A = []
        # self.images_B = []
        # self.images_label = []

    def __getitem__(self, index):
        image_A = np.asarray(Image.open(self.imageA_path[index]).convert("RGB"))  # 0 ~ 255 shape:h,w,c
        image_B = np.asarray(Image.open(self.imageB_path[index]).convert("RGB"))  # 0 ~ 255 shape:h,w,c
        image_Label = np.array(Image.open(self.imageLabel_path[index]), dtype=np.uint8)  # shape:h,w
        image_Label = image_Label // 255  # 把标签转化为0，1标签

        if self.transform:
            # image_A : 3, 256, 256  image_Label: 1, 256, 256    (0-255)
            [image_A, image_B], [image_Label] = self.augmentation.transform([image_A, image_B], [image_Label],
                                                                            to_tensor_and_norm=True)

        return {"image_A": image_A, "image_B": image_B, "image_Label": image_Label}

    def __len__(self):
        return len(self.imageA_path)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
