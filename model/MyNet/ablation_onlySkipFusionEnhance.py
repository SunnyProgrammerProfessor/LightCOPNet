# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    :ablation_DMCConv.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/5/20 18:27 
"""
import sys
sys.path.append("/home/data/b532zhaoxiaohui/shuaige/ChangeDetection")
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MyNet.net_component_ablation_onlySkipFusionEnhance import BasicConv, Encoder_Block, Differ_Enhance, Decoder_Block
from thop import profile


class MyCDNet(nn.Module):
    def __init__(self):
        super(MyCDNet, self).__init__()
        # Encoder
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # 8,32,256,256
        self.encoder_block1_basicconv1 = BasicConv(in_channels=32, out_channels=32)
        self.encoder_block1_basicconv2 = BasicConv(in_channels=32, out_channels=32)

        self.encoder_block2 = Encoder_Block(in_channels=32, out_channels=64)  # 8,64,128,128
        self.differ_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # self.en2_x2_spectrumspatialenhance = Differ_Enhance(out_channels=64)
        self.seg1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True)
        )  # 8,1,128,128
        self.encoder_block3 = Encoder_Block(in_channels=64, out_channels=128)  # 8,128,64,64
        self.differ_2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # self.en3_x2_spectrumspatialenhance = Differ_Enhance(out_channels=128)
        self.seg2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True)
        )  # 8,1,64,64
        self.encoder_block4 = Encoder_Block(in_channels=128, out_channels=256)  # 8,256,32,32
        self.differ_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # self.en4_x2_spectrumspatialenhance = Differ_Enhance(out_channels=256)
        self.seg3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True)
        )  # 8,1,32,32
        self.encoder_block5 = Encoder_Block(in_channels=256, out_channels=512)  # 8,512,16,16
        self.differ_4_conv = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.decoder_block1 = Decoder_Block(in_channels=512, out_channels=256)  # 8,256,32,32
        self.decoder_block2 = Decoder_Block(in_channels=256, out_channels=128)  # 8,128,64,64
        self.decoder_block3 = Decoder_Block(in_channels=128, out_channels=64)  # 8,64,128,128
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.change_out = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x1, x2):
        # encoder
        x1_en1 = self.encoder_block1(x1)  # 8,32,256,256
        x1_en1 = self.encoder_block1_basicconv2(self.encoder_block1_basicconv1(x1_en1))  # 8,32,256,256
        x2_en1 = self.encoder_block1(x2)  # 8,32,256,256
        x2_en1 = self.encoder_block1_basicconv2(self.encoder_block1_basicconv1(x2_en1))  # 8,32,256,256
        x1_en2 = self.encoder_block2(x1_en1)  # 8,64,128,128
        x2_en2 = self.encoder_block2(x2_en1)  # 8,64,128,128
        differ_1 = self.differ_1_conv(torch.abs(x2_en2 - x1_en2))  # 8,64,128,128
        # x2_en2 = self.en2_x2_spectrumspatialenhance(x2_en2, differ_1)  # 8,64,128,128
        # differ_1_out = F.interpolate(differ_1,scale_factor=(2,2),mode="bilinear")# 8,64,256,256
        # differ_1_out = self.seg1(differ_1_out)  # 8,1,256,256
        x1_en3 = self.encoder_block3(x1_en2)  # 8,128,64,64
        x2_en3 = self.encoder_block3(x2_en2)  # 8,128,64,64
        differ_2 = self.differ_2_conv(torch.abs(x2_en3 - x1_en3))  # 8,128,64,64
        # x2_en3 = self.en3_x2_spectrumspatialenhance(x2_en3, differ_2)  # 8,128,64,64
        # differ_2_out = F.interpolate(differ_2,scale_factor=(4,4),mode="bilinear")# 8,128,256,256
        # differ_2_out = self.seg2(differ_2_out)  # 8,1,256,256
        x1_en4 = self.encoder_block4(x1_en3)  # 8,256,32,32
        x2_en4 = self.encoder_block4(x2_en3)  # 8,256,32,32
        differ_3 = self.differ_3_conv(torch.abs(x2_en4 - x1_en4))  # 8,256,32,32
        # x2_en4 = self.en4_x2_spectrumspatialenhance(differ_3, x2_en4)  # 8,256,32,32
        # differ_3_out = F.interpolate(differ_3,scale_factor=(8,8),mode="bilinear")# 8,256,256,256
        # differ_3_out = self.seg3(differ_3_out)  # 8,1,256,256
        x1_en5 = self.encoder_block5(x1_en4)  # 8,512,16,16
        x2_en5 = self.encoder_block5(x2_en4)  # 8,512,16,16
        differ_4 = self.differ_4_conv(torch.abs(x2_en5 - x1_en5))  # 8,512,16,16
        # decoder
        differ_de1 = self.decoder_block1(differ_3, differ_4)  # 8,256,32,32
        differ_de2 = self.decoder_block2(differ_2, differ_de1)  # 8,128,64,64
        differ_de3 = self.decoder_block3(differ_1, differ_de2)  # 8,64,128,128
        change_out = self.upsample(differ_de3)
        out = self.change_out(change_out)  # 8,1,256,256

        # return differ_1_out, differ_2_out, differ_3_out, out
        return out
    
# net = MyCDNet().to("cuda:3")
# T1 = torch.randn(8,3,256,256).to("cuda:3")
# T2 = torch.randn(8,3,256,256).to("cuda:3")
# f,p = profile(model=net,inputs=(T1,T2))
# print(p/1e6)