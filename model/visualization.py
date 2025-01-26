# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：visualization.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/3/24 20:56 
"""
from torchvision import utils
import numpy as np


def make_numpy_grid(tensor, pad_value=0, padding=0):  # tensor:8,3,256,256
    tensor = tensor.detach()
    vis = utils.make_grid(tensor, pad_value=pad_value, padding=padding)  # vis:3,256,2048
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def vis_norm(tensor):
    return tensor * 0.5 + 0.5
