# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/4/28 20:19 
"""
import sys
sys.path.append("/home/data/b532zhaoxiaohui/shuaige/ChangeDetection")
import os
import os.path
import random
import glob

import matplotlib.pyplot as plt
import torch

# image_path = "F:\\BIT_CD-master\\BIT_CD-master\\LEVIR"
# image_filename = os.path.basename(image_path)
# path_list = os.listdir(image_path)

imgA_folder = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/CDD/A"
imgB_folder = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/CDD/B"
imgLabel_folder = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/CDD/label"
data_forms = ["train", "val", "test"]
# image_base_folder = "F:\\ChangeDetection\\LEVIR\\list"


# 将图像的名称存储在txt文件中
# def read_image_name(root: str, form):  # root : 数据集的图像所在的路径, forms : 保存数据集名称的类型(train,val,test)
#     imageLists = glob.glob(root)
#     imageLists.sort()
#     with open(r"F:\\ChangeDetection\\LEVIR\\list\\train.txt", "a", encoding="utf-8") as f:
#         for img in imageLists:
#             image_name = img.split("/")[-1]
#             image_name_ = image_name.split("\\")[-1]
#             f.write(image_name_ + "\n")
#     f.close()


def read_split_data(root: str, split_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可以复现
    assert os.path.exists(root), f"Your dataset {root} does not exist"
    suffix = ["png", "PNG", "JPG", "jpg"]
    train_dataset, train_label, test_dataset, test_label = [], [], [], []


# 获取训练数据集
def get_image_data(form: str, path: str):
    form += ".txt"
    path = os.path.join(path, form)
    levir_A = []
    levir_B = []
    levir_label = []
    with open(path) as f:
        image_names = f.readlines()
        for i in image_names:
            levir_A_path = os.path.join(imgA_folder, i)
            levir_A.append(levir_A_path[0:-1])
            levir_B_path = os.path.join(imgB_folder, i)
            levir_B.append(levir_B_path[0:-1])
            levir_label_path = os.path.join(imgLabel_folder, i)
            levir_label.append(levir_label_path[0:-1])
    f.close()
    return levir_A, levir_B, levir_label


# read_split_data(os.path.join(image_path, "A"), 0.2)
# print(torch.cuda.get_device_capability(device=0))
# form = "train"
# a, b, c = get_image_data(form=form)
# print(c)
# with open("F:\\ChangeDetection\\LEVIR\\list\\train.txt") as f:
#     image_names = f.readlines()
#     print(image_names)
#     f.close()
