# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学
@File    ：net_utils.py
@IDE     ：PyCharm
@Author  ：崔俊贤
@Date    ：2024/3/17 22:10
"""
import torch
import torch.nn as nn
from torch.nn import init


# 神经网络初始化
def init_net(net, gpu_ids, device, init_type="normal", init_gain=0.02):
    assert (torch.cuda.is_available())
    if len(gpu_ids) > 0:
        net.to(device)
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain)
    return net


# 神经网络参数初始化
def init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(model):
        classname = model.__class__.__name__
        if hasattr(model, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(model.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(model.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(model.weight.data, 0, "fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(model.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"initialization method {init_type} is not implemented")
        if hasattr(model, "bias") and model.bias is not None:
            init.constant_(model.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(model.weight.data, 1.0, init_gain)
            init.constant_(model.bias.data, 0.0)

    print(f"Initialize network with {init_type}")
    net.apply(init_func)  # net apply this initialization
