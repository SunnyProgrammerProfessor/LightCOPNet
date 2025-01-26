# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：loss.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/3/22 17:11 
"""
import torch
import torch.nn.functional as F


# cross_entropy
def cross_entropy(input, target, weight=None, reduction="mean", ignore_index=255):
    target = target.long()
    if target.dim() == 4:  # target : NCHW - > NHW
        target = torch.squeeze(target, dim=1)
    if target.shape[-1] != input.shape[-1]:  # 如果模型输出结果和标签大小不一样，就将结果进行上采样
        input = F.interpolate(input, size=target.shape[1:], mode="bilinear", align_corners=True)
    return F.cross_entropy(input=input, target=target, weight=weight, ignore_index=ignore_index,
                           reduction=reduction)


# binary_ce
# def binary_ce(input, target, weight=None, reduction='mean'):
#     # input:out(8,1,256,256),x1_seg(8,1,128,128),x2_seg(8,1,128,128)
#     # target:image_gt:(8,1,256,256) , seg_gt:(8,1,128,128)
#     target = [t.float() for t in target]
#     target = [t.squeeze(1) if len(t.shape) > 3 else t for t in target]
#     input = [i.squeeze(1).float() if len(i.shape) > 3 else i for i in input]
#     return F.binary_cross_entropy(input=torch.sigmoid(input[0]), target=target[0], weight=weight, reduction=reduction) \
#            + 0.2 * (F.binary_cross_entropy(input=torch.sigmoid(input[1]), target=target[1], weight=weight,
#                                            reduction=reduction)
#                     + F.binary_cross_entropy(input=torch.sigmoid(input[2]), target=target[1], weight=weight,
#                                              reduction=reduction))

# def binary_ce(input, target, weight=None, reduction='mean'):
#     # input:out(8,1,256,256),seg_pred1(8,1,128,128),seg_pred2(8,1,64,64),seg_pred3(8,1,32,32)
#     # target:image_gt:(8,1,256,256) , seg_gt1(8,1,128,128),seg_gt2(8,1,64,64),seg_gt3(8,1,32,32)
#     target = [t.float() for t in target]
#     target = [t.squeeze(1) if len(t.shape) > 3 else t for t in target]
#     input = [i.squeeze(1).float() if len(i.shape) > 3 else i for i in input]
#     return F.binary_cross_entropy(input=torch.sigmoid(input[3]), target=target[0], weight=weight,
#                                   reduction=reduction) + 0.2 * (
#                    F.binary_cross_entropy(input=torch.sigmoid(input[0]), target=target[1], weight=weight,
#                                           reduction=reduction) + F.binary_cross_entropy(
#                input=torch.sigmoid(input[1]), target=target[2], weight=weight,
#                reduction=reduction) + F.binary_cross_entropy(input=torch.sigmoid(input[2]), target=target[3],
#                                                              weight=weight, reduction=reduction))

# # 只计算最终损失
def binary_ce(input, target, weight=None, reduction='mean'):
    # input:out(8,1,256,256),seg_pred1(8,1,128,128),seg_pred2(8,1,64,64),seg_pred3(8,1,32,32)
    # target:image_gt:(8,1,256,256) , seg_gt1(8,1,128,128),seg_gt2(8,1,64,64),seg_gt3(8,1,32,32)
    target = [t.float() for t in target]
    target = [t.squeeze(1) if len(t.shape) > 3 else t for t in target]
    input = input.squeeze(1).float() if len(input.shape) > 3 else input 
    return F.binary_cross_entropy(input=torch.sigmoid(input), target=target[0], weight=weight, reduction=reduction)
