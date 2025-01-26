from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# 可视化结果保存路径
visualization_CDD = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/visualization_CDD"

# 加载图片
# UCPE-Net
image_CDD_ours = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_203_batch_1_.jpg"
image_CDD_ours = Image.open(image_CDD_ours)
# TransUNetCD
image_CDD_TransUNetCD = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_151_batch_1_.jpg" 
image_CDD_TransUNetCD = Image.open(image_CDD_TransUNetCD)
# mamba
image_CDD_mamba = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_143_batch_1_.jpg" 
image_CDD_mamba = Image.open(image_CDD_mamba)
# T_UNet
image_CDD_T_UNet = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_150_batch_1_.jpg" 
image_CDD_T_UNet = Image.open(image_CDD_T_UNet)
# ELGC
image_CDD_ELGC = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_191_batch_1_.jpg" 
image_CDD_ELGC = Image.open(image_CDD_ELGC)
# BiT
image_CDD_BiT = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_413_batch_1_.jpg" 
image_CDD_BiT = Image.open(image_CDD_BiT)
# USSFC
image_CDD_USSFC = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_248_batch_1_.jpg" 
image_CDD_USSFC = Image.open(image_CDD_USSFC)
# DSAMNet
image_CDD_DSAMNet = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_43_batch_1_.jpg" 
image_CDD_DSAMNet = Image.open(image_CDD_DSAMNet)
# VcT
image_CDD_VcT = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_35_batch_1_.jpg" 
image_CDD_VcT = Image.open(image_CDD_VcT)
# DTCDSCN
image_CDD_DTCDSCN = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_84_batch_1_.jpg" 
image_CDD_DTCDSCN = Image.open(image_CDD_DTCDSCN)
# DASNet
image_CDD_DASNet = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_465_batch_1_.jpg" 
image_CDD_DASNet = Image.open(image_CDD_DASNet)
# SNUNet
image_CDD_SNUNet = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_494_batch_1_.jpg" 
image_CDD_SNUNet = Image.open(image_CDD_SNUNet)
# ChangeFormer
image_CDD_ChangeFormer = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_483_batch_1_.jpg" 
image_CDD_ChangeFormer = Image.open(image_CDD_ChangeFormer)
# DESSN
image_CDD_DESSN = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_20_batch_1_.jpg" 
image_CDD_DESSN = Image.open(image_CDD_DESSN)
# STANet
image_CDD_STANet = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_234_batch_1_.jpg" 
image_CDD_STANet = Image.open(image_CDD_STANet)
# FCN_PP
image_CDD_FCN_PP = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_3_batch_1_.jpg" 
image_CDD_FCN_PP = Image.open(image_CDD_FCN_PP)
# IFNet
image_CDD_IFNet = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_69_batch_1_.jpg" 
image_CDD_IFNet = Image.open(image_CDD_IFNet)
# FC_EF
image_CDD_FC_EF = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_87_batch_1_.jpg" 
image_CDD_FC_EF = Image.open(image_CDD_FC_EF)
# FC_Siam_Diff
image_CDD_FC_Siam_Diff = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_1_batch_1_.jpg" 
image_CDD_FC_Siam_Diff = Image.open(image_CDD_FC_Siam_Diff)
# FC_Siam_Conc
image_CDD_FC_Siam_Conc = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_12_batch_1_.jpg" 
image_CDD_FC_Siam_Conc = Image.open(image_CDD_FC_Siam_Conc)
# T1
image_CDD_T1 = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_203_batch_1_.jpg"
image_CDD_T1 = Image.open(image_CDD_T1)
# T2
image_CDD_T2 = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_203_batch_1_.jpg"
image_CDD_T2 = Image.open(image_CDD_T2)
# gt
image_CDD_gt = "/home/data/b532zhaoxiaohui/shuaige/ChangeDetection/vis/MyNet/CDD/isTrain_False-epoch_203_batch_1_.jpg"
image_CDD_gt = Image.open(image_CDD_gt)

# 转化为Tensor
transform = transforms.ToTensor()  # c,h,w
# UCPE-Net
image_ours = transform(image_CDD_ours)
#RS-Mamba
image_mamba = transform(image_CDD_mamba)
# VcT
image_VcT = transform(image_CDD_VcT)
# ELGC-Net
image_ELGC = transform(image_CDD_ELGC)
# USSFC
image_USSFC = transform(image_CDD_USSFC)
# FC_EF
image_FC_EF = transform(image_CDD_FC_EF)
# FC_Siam_Diff
image_FC_Siam_Diff = transform(image_CDD_FC_Siam_Diff)
# FC_Siam_Conc
image_FC_Siam_Conc = transform(image_CDD_FC_Siam_Conc)
# DTCDSCN
image_DTCDSCN = transform(image_CDD_DTCDSCN)
# IFNet
image_IFNet = transform(image_CDD_IFNet)
# BiT
image_BiT = transform(image_CDD_BiT)
# SNUNet
image_SNUNet = transform(image_CDD_SNUNet)
# TransUNetCD
image_TransUNetCD = transform(image_CDD_TransUNetCD)
# STANet
image_STANet = transform(image_CDD_STANet)
# ChangeFormer
image_ChangeFormer = transform(image_CDD_ChangeFormer)
# DASNet
image_DASNet = transform(image_CDD_DASNet)
# DESSN
image_DESSN = transform(image_CDD_DESSN)
# T_UNet
image_T_UNet = transform(image_CDD_T_UNet)
# DSAMNet
image_DSAMNet = transform(image_CDD_DSAMNet)
# FCN_PP
image_FCN_PP = transform(image_CDD_FCN_PP)
# T
image_T1 = transform(image_CDD_T1)
# T2
image_T2 = transform(image_CDD_T2)
# gt
image_gt = transform(image_CDD_gt)

# 定义裁剪的高度和宽度
crop_height,crop_width = 256,256

# 计算裁剪区域的起始位置
# ===================first==============================
ours_x1 = 1280
ours_y1 = 2304
# T11
T1_x1 = 1280
T1_y1 = 256
# T21
T2_x1 = 1280
T2_y1 = 768
# gt1
gt_x1 = 1280
gt_y1 = 1792
# ===================first==============================

# ===================second==============================
ours_x2 = 0
ours_y2 = 2048
# T12
T1_x2 = 0
T1_y2 = 0
# T22
T2_x2 = 0
T2_y2 = 512
# gt2
gt_x2 = 0
gt_y2 = 1536
# ===================second==============================

# ===================third==============================
ours_x3 = 768
ours_y3 = 2048
# T13
T1_x3 = 768
T1_y3 = 0
# T23
T2_x3 = 768
T2_y3 = 512
# gt3
gt_x3 = 768
gt_y3 = 1536
# ===================third==============================

# ===================forth==============================
ours_x4 = 768
ours_y4 = 2304
# T14
T1_x4 = 768
T1_y4 = 256
# T24
T2_x4 = 768
T2_y4 = 768
# gt4
gt_x4 = 768
gt_y4 = 1792
# ===================forth==============================

# 可视化结果中间加入间隔
interval_w = torch.ones(3,256,3)
interval_h = torch.ones(3,5,5954)

# 裁剪的图片
# ===================first==============================
# UCPE-Net
cropped_tensor_ours1 = image_ours[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
#RS-Mamba
cropped_tensor_mamba1 = image_mamba[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
#VcT
cropped_tensor_VcT1 = image_VcT[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# ELGC-Net
cropped_tensor_ELGC1 = image_ELGC[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# USSFC
cropped_tensor_USSFC1 = image_USSFC[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# FC_EF
cropped_tensor_FC_EF1 = image_FC_EF[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# FC_Siam_Diff
cropped_tensor_FC_Siam_Diff1 = image_FC_Siam_Diff[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# FC_Siam_Conc
cropped_tensor_FC_Siam_Conc1 = image_FC_Siam_Conc[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# DTCDSCN
cropped_tensor_DTCDSCN1 = image_DTCDSCN[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# IFNet
cropped_tensor_IFNet1 = image_IFNet[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# BiT
cropped_tensor_BiT1 = image_BiT[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# SNUNet
cropped_tensor_SNUNet1 = image_SNUNet[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# TransUNetCD
cropped_tensor_TransUNetCD1 = image_TransUNetCD[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# STANet
cropped_tensor_STANet1 = image_STANet[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# ChangeFormer
cropped_tensor_ChangeFormer1 = image_ChangeFormer[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# DASNet
cropped_tensor_DASNet1 = image_DASNet[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# DESSN
cropped_tensor_DESSN1 = image_DESSN[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# T_UNet
cropped_tensor_T_UNet1 = image_T_UNet[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# DSAMNet
cropped_tensor_DSAMNet1 = image_DSAMNet[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# FCN_PP
cropped_tensor_FCN_PP1 = image_FCN_PP[:,ours_y1:ours_y1+crop_height,ours_x1:ours_x1+crop_width]
# T1
cropped_tensor_T11 = image_T1[:,T1_y1:T1_y1+crop_height,T1_x1:T1_x1+crop_width]
# T2
cropped_tensor_T21 = image_T2[:,T2_y1:T2_y1+crop_height,T2_x1:T2_x1+crop_width]
# gt
cropped_tensor_gt1 = image_gt[:,gt_y1:gt_y1+crop_height,gt_x1:gt_x1+crop_width]
# ===================first==============================

# ===================second==============================
# UCPE-Net
cropped_tensor_ours2 = image_ours[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
#RS-Mamba
cropped_tensor_mamba2 = image_mamba[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
#VcT
cropped_tensor_VcT2 = image_VcT[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# ELGC-Net
cropped_tensor_ELGC2 = image_ELGC[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# USSFC
cropped_tensor_USSFC2 = image_USSFC[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# FC_EF
cropped_tensor_FC_EF2 = image_FC_EF[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# FC_Siam_Diff
cropped_tensor_FC_Siam_Diff2 = image_FC_Siam_Diff[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# FC_Siam_Conc
cropped_tensor_FC_Siam_Conc2 = image_FC_Siam_Conc[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# DTCDSCN
cropped_tensor_DTCDSCN2 = image_DTCDSCN[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# IFNet
cropped_tensor_IFNet2 = image_IFNet[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# BiT
cropped_tensor_BiT2 = image_BiT[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# SNUNet
cropped_tensor_SNUNet2 = image_SNUNet[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# TransUNetCD
cropped_tensor_TransUNetCD2 = image_TransUNetCD[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# STANet
cropped_tensor_STANet2 = image_STANet[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# ChangeFormer
cropped_tensor_ChangeFormer2 = image_ChangeFormer[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# DASNet
cropped_tensor_DASNet2 = image_DASNet[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# DESSN
cropped_tensor_DESSN2 = image_DESSN[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# T_UNet
cropped_tensor_T_UNet2 = image_T_UNet[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# DSAMNet
cropped_tensor_DSAMNet2 = image_DSAMNet[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# FCN_PP
cropped_tensor_FCN_PP2 = image_FCN_PP[:,ours_y2:ours_y2+crop_height,ours_x2:ours_x2+crop_width]
# T1
cropped_tensor_T12 = image_T1[:,T1_y2:T1_y2+crop_height,T1_x2:T1_x2+crop_width]
# T2
cropped_tensor_T22 = image_T2[:,T2_y2:T2_y2+crop_height,T2_x2:T2_x2+crop_width]
# gt
cropped_tensor_gt2 = image_gt[:,gt_y2:gt_y2+crop_height,gt_x2:gt_x2+crop_width]
# ===================second==============================

# ===================third==============================
# UCPE-Net
cropped_tensor_ours3 = image_ours[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
#RS-Mamba
cropped_tensor_mamba3 = image_mamba[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
#VcT
cropped_tensor_VcT3 = image_VcT[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# ELGC-Net
cropped_tensor_ELGC3 = image_ELGC[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# USSFC
cropped_tensor_USSFC3 = image_USSFC[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# FC_EF
cropped_tensor_FC_EF3 = image_FC_EF[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# FC_Siam_Diff
cropped_tensor_FC_Siam_Diff3 = image_FC_Siam_Diff[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# FC_Siam_Conc
cropped_tensor_FC_Siam_Conc3 = image_FC_Siam_Conc[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# DTCDSCN
cropped_tensor_DTCDSCN3 = image_DTCDSCN[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# IFNet
cropped_tensor_IFNet3 = image_IFNet[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# BiT
cropped_tensor_BiT3 = image_BiT[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# SNUNet
cropped_tensor_SNUNet3 = image_SNUNet[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# TransUNetCD
cropped_tensor_TransUNetCD3 = image_TransUNetCD[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# STANet
cropped_tensor_STANet3 = image_STANet[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# ChangeFormer
cropped_tensor_ChangeFormer3 = image_ChangeFormer[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# DASNet
cropped_tensor_DASNet3 = image_DASNet[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# DESSN
cropped_tensor_DESSN3 = image_DESSN[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# T_UNet
cropped_tensor_T_UNet3 = image_T_UNet[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# DSAMNet
cropped_tensor_DSAMNet3 = image_DSAMNet[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# FCN_PP
cropped_tensor_FCN_PP3 = image_FCN_PP[:,ours_y3:ours_y3+crop_height,ours_x3:ours_x3+crop_width]
# T1
cropped_tensor_T13 = image_T1[:,T1_y3:T1_y3+crop_height,T1_x3:T1_x3+crop_width]
# T2
cropped_tensor_T23 = image_T2[:,T2_y3:T2_y3+crop_height,T2_x3:T2_x3+crop_width]
# gt
cropped_tensor_gt3 = image_gt[:,gt_y3:gt_y3+crop_height,gt_x3:gt_x3+crop_width]
# ===================third==============================

# ===================forth==============================
# UCPE-Net
cropped_tensor_ours4 = image_ours[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
#RS-Mamba
cropped_tensor_mamba4 = image_mamba[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
#VcT
cropped_tensor_VcT4 = image_VcT[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# ELGC-Net
cropped_tensor_ELGC4 = image_ELGC[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# USSFC
cropped_tensor_USSFC4 = image_USSFC[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# FC_EF
cropped_tensor_FC_EF4 = image_FC_EF[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# FC_Siam_Diff
cropped_tensor_FC_Siam_Diff4 = image_FC_Siam_Diff[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# FC_Siam_Conc
cropped_tensor_FC_Siam_Conc4 = image_FC_Siam_Conc[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# DTCDSCN
cropped_tensor_DTCDSCN4 = image_DTCDSCN[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# IFNet
cropped_tensor_IFNet4 = image_IFNet[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# BiT
cropped_tensor_BiT4 = image_BiT[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# SNUNet
cropped_tensor_SNUNet4 = image_SNUNet[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# TransUNetCD
cropped_tensor_TransUNetCD4 = image_TransUNetCD[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# STANet
cropped_tensor_STANet4 = image_STANet[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# ChangeFormer
cropped_tensor_ChangeFormer4 = image_ChangeFormer[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# DASNet
cropped_tensor_DASNet4 = image_DASNet[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# DESSN
cropped_tensor_DESSN4 = image_DESSN[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# T_UNet
cropped_tensor_T_UNet4 = image_T_UNet[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# DSAMNet
cropped_tensor_DSAMNet4 = image_DSAMNet[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# FCN_PP
cropped_tensor_FCN_PP4 = image_FCN_PP[:,ours_y4:ours_y4+crop_height,ours_x4:ours_x4+crop_width]
# T1
cropped_tensor_T14 = image_T1[:,T1_y4:T1_y4+crop_height,T1_x4:T1_x4+crop_width]
# T2
cropped_tensor_T24 = image_T2[:,T2_y4:T2_y4+crop_height,T2_x4:T2_x4+crop_width]
# gt
cropped_tensor_gt4 = image_gt[:,gt_y4:gt_y4+crop_height,gt_x4:gt_x4+crop_width]
# ===================forth==============================

# ===================first==============================
cropped_tensor_combine1 = torch.cat(
    [cropped_tensor_T11,interval_w,cropped_tensor_T21,interval_w,cropped_tensor_gt1,interval_w,
     cropped_tensor_FC_EF1,interval_w,cropped_tensor_FC_Siam_Diff1,interval_w,cropped_tensor_FC_Siam_Conc1,interval_w,
     cropped_tensor_FCN_PP1,interval_w,cropped_tensor_DTCDSCN1,interval_w,cropped_tensor_IFNet1,interval_w,
     cropped_tensor_STANet1,interval_w,cropped_tensor_DASNet1,interval_w,cropped_tensor_BiT1,interval_w,
     cropped_tensor_DESSN1,interval_w,cropped_tensor_SNUNet1,interval_w,cropped_tensor_DSAMNet1,interval_w,
     cropped_tensor_TransUNetCD1,interval_w,cropped_tensor_ChangeFormer1,interval_w,cropped_tensor_USSFC1,interval_w,
     cropped_tensor_VcT1,interval_w,cropped_tensor_ELGC1,interval_w,cropped_tensor_T_UNet1,interval_w,
     cropped_tensor_mamba1,interval_w,cropped_tensor_ours1
    ],dim=2
)
# ===================first==============================

# ===================second==============================
cropped_tensor_combine2 = torch.cat(
    [cropped_tensor_T12,interval_w,cropped_tensor_T22,interval_w,cropped_tensor_gt2,interval_w,
     cropped_tensor_FC_EF2,interval_w,cropped_tensor_FC_Siam_Diff2,interval_w,cropped_tensor_FC_Siam_Conc2,interval_w,
     cropped_tensor_FCN_PP2,interval_w,cropped_tensor_DTCDSCN2,interval_w,cropped_tensor_IFNet2,interval_w,
     cropped_tensor_STANet2,interval_w,cropped_tensor_DASNet2,interval_w,cropped_tensor_BiT2,interval_w,
     cropped_tensor_DESSN2,interval_w,cropped_tensor_SNUNet2,interval_w,cropped_tensor_DSAMNet2,interval_w,
     cropped_tensor_TransUNetCD2,interval_w,cropped_tensor_ChangeFormer2,interval_w,cropped_tensor_USSFC2,interval_w,
     cropped_tensor_VcT2,interval_w,cropped_tensor_ELGC2,interval_w,cropped_tensor_T_UNet2,interval_w,
     cropped_tensor_mamba2,interval_w,cropped_tensor_ours2
    ],dim=2
)
# ===================second==============================

# ===================third==============================
cropped_tensor_combine3 = torch.cat(
    [cropped_tensor_T13,interval_w,cropped_tensor_T23,interval_w,cropped_tensor_gt3,interval_w,
     cropped_tensor_FC_EF3,interval_w,cropped_tensor_FC_Siam_Diff3,interval_w,cropped_tensor_FC_Siam_Conc3,interval_w,
     cropped_tensor_FCN_PP3,interval_w,cropped_tensor_DTCDSCN3,interval_w,cropped_tensor_IFNet3,interval_w,
     cropped_tensor_STANet3,interval_w,cropped_tensor_DASNet3,interval_w,cropped_tensor_BiT3,interval_w,
     cropped_tensor_DESSN3,interval_w,cropped_tensor_SNUNet3,interval_w,cropped_tensor_DSAMNet3,interval_w,
     cropped_tensor_TransUNetCD3,interval_w,cropped_tensor_ChangeFormer3,interval_w,cropped_tensor_USSFC3,interval_w,
     cropped_tensor_VcT3,interval_w,cropped_tensor_ELGC3,interval_w,cropped_tensor_T_UNet3,interval_w,
     cropped_tensor_mamba3,interval_w,cropped_tensor_ours3
    ],dim=2
)
# ===================third==============================

# ===================forth==============================
cropped_tensor_combine4 = torch.cat(
    [cropped_tensor_T14,interval_w,cropped_tensor_T24,interval_w,cropped_tensor_gt4,interval_w,
     cropped_tensor_FC_EF4,interval_w,cropped_tensor_FC_Siam_Diff4,interval_w,cropped_tensor_FC_Siam_Conc4,interval_w,
     cropped_tensor_FCN_PP4,interval_w,cropped_tensor_DTCDSCN4,interval_w,cropped_tensor_IFNet4,interval_w,
     cropped_tensor_STANet4,interval_w,cropped_tensor_DASNet4,interval_w,cropped_tensor_BiT4,interval_w,
     cropped_tensor_DESSN4,interval_w,cropped_tensor_SNUNet4,interval_w,cropped_tensor_DSAMNet4,interval_w,
     cropped_tensor_TransUNetCD4,interval_w,cropped_tensor_ChangeFormer4,interval_w,cropped_tensor_USSFC4,interval_w,
     cropped_tensor_VcT4,interval_w,cropped_tensor_ELGC4,interval_w,cropped_tensor_T_UNet4,interval_w,
     cropped_tensor_mamba4,interval_w,cropped_tensor_ours4
    ],dim=2
)
# ===================forth==============================

# Dual-Image
visualization1 = transforms.ToPILImage()(cropped_tensor_combine1)
visualization1.save(os.path.join(visualization_CDD,"visualization_comparison1.png"))
visualization2 = transforms.ToPILImage()(cropped_tensor_combine2)
visualization2.save(os.path.join(visualization_CDD,"visualization_comparison2.png"))
visualization3 = transforms.ToPILImage()(cropped_tensor_combine3)
visualization3.save(os.path.join(visualization_CDD,"visualization_comparison3.png"))
visualization4 = transforms.ToPILImage()(cropped_tensor_combine4)
visualization4.save(os.path.join(visualization_CDD,"visualization_comparison4.png"))

# combine
visualization_tensor_combine = torch.cat([cropped_tensor_combine1,interval_h,cropped_tensor_combine2,interval_h,
                                   cropped_tensor_combine3,interval_h,cropped_tensor_combine4],dim=1)
visualization_combine = transforms.ToPILImage()(visualization_tensor_combine)
visualization_combine.save(os.path.join(visualization_CDD,"visualization_combine.png"))








