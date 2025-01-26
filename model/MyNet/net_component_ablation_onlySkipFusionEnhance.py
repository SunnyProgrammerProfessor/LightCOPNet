# -*- coding: UTF-8 -*-
"""
@Project ：吉林大学 
@File    ：net_component.py
@IDE     ：PyCharm 
@Author  ：崔俊贤
@Date    ：2024/5/20 18:01 
"""
import torch
import torch.nn as nn
import math


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=1,kernel_size=3,padding=1)

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        out = self.bn_relu(x)
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=2, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2, x3 = split_channel(x)
        x2 = self.max_pool(x2)
        x3 = self.avg_pool(x3)
        out_max_avg_pool = torch.cat([x2, x3], dim=1)
        out = torch.cat([x1, out_max_avg_pool], dim=1)
        return out


class CSAM(nn.Module):
    def __init__(self, out_channels):
        super(CSAM, self).__init__()
        self.k = kernel_size(out_channels)
        self.channel_avg_attention = nn.AdaptiveAvgPool2d(1)  # b,c,1,1
        self.channel_max_attention = nn.AdaptiveMaxPool2d(1)  # b,c,1,1
        self.channel_attention = nn.Conv1d(
            in_channels=2, out_channels=1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_attention = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_avg_attention = self.channel_avg_attention(
            x).squeeze(-1).permute(0, 2, 1)  # b,1,c
        channel_max_attention = self.channel_max_attention(
            x).squeeze(-1).permute(0, 2, 1)  # b,1,c
        channel_attention = self.channel_attention(
            torch.cat([channel_max_attention, channel_avg_attention], dim=1))  # b,1,c
        channel_attention = channel_attention.permute(
            0, 2, 1).unsqueeze(-1)  # b,c,1,1
        channel_attention = self.sigmoid(channel_attention)
        x = x * channel_attention
        spatial_max_attention = torch.max(x, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_avg_attention = torch.mean(x, dim=1, keepdim=True)  # b,1,h,w
        spatial_attention = self.spatial_attention(
            torch.cat([spatial_max_attention, spatial_avg_attention], dim=1))  # b,1,h,w
        spatial_attention = self.sigmoid(spatial_attention)
        out = x * spatial_attention  # b,c,h,w
        return out


class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_Block, self).__init__()
        self.pre_basic_conv = DownSample(
            in_channels=in_channels, out_channels=out_channels)
        self.conv1 = BasicConv(in_channels=out_channels,
                               out_channels=out_channels)
        self.conv2 = BasicConv(in_channels=out_channels,
                               out_channels=out_channels)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.csam = CSAM(out_channels=out_channels)

    def forward(self, x):
        x = self.pre_basic_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_bn_relu(x)
        residual = x.clone()
        x = self.csam(x)
        out = residual + x
        return out


class Differ_Enhance(nn.Module):
    def __init__(self, out_channels):
        super(Differ_Enhance, self).__init__()
        self.channel_avg_attention = nn.AdaptiveAvgPool2d(1)  # b,c,1,1
        self.channel_max_attention = nn.AdaptiveMaxPool2d(1)  # b,c,1,1
        self.k = kernel_size(out_channels)
        self.channel_attention = nn.Conv1d(
            in_channels=2, out_channels=1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_attention = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x2, differ):
        x2_residual = x2.clone()
        channel_avg_attention = self.channel_avg_attention(
            differ).squeeze(-1).permute(0, 2, 1)  # b,1,c
        channel_max_attention = self.channel_max_attention(
            differ).squeeze(-1).permute(0, 2, 1)  # b,1,c
        channel_attention = self.channel_attention(
            torch.cat([channel_max_attention, channel_avg_attention], dim=1)).permute(0, 2, 1).unsqueeze(-1)  # b,c,1,1
        channel_attention = self.sigmoid(channel_attention)
        x2 = x2 * channel_attention
        spatial_max_attention = torch.max(
            differ, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_avg_attention = torch.mean(
            differ, dim=1, keepdim=True)  # b,1,h,w
        spatial_attention = self.spatial_attention(
            torch.cat([spatial_max_attention, spatial_avg_attention], dim=1))  # b,1,h,w
        spatial_attention = self.sigmoid(spatial_attention)
        out = x2 * spatial_attention + x2_residual
        return out


class Exchange_Encoder_Decoder_Channel(nn.Module):
    def __init__(self, interval=2):
        super(Exchange_Encoder_Decoder_Channel, self).__init__()
        self.interval = interval

    def forward(self, x1, x2):
        batch_size, num_channels, height, weight = x1.shape
        exchange_mask = (torch.arange(num_channels) %
                         self.interval == 0).to("cuda:3").int()
        exchange_mask1 = exchange_mask.expand(
            batch_size, num_channels).unsqueeze(-1).unsqueeze(-1)
        exchange_mask2 = 1 - exchange_mask1
        x1 = x1 * exchange_mask1 + x2 * exchange_mask2
        x2 = x2 * exchange_mask1 + x1 * exchange_mask2
        return x1, x2


class Skip_Fusion_Enhance(nn.Module):
    def __init__(self, channels):
        super(Skip_Fusion_Enhance, self).__init__()
        self.k = kernel_size(channels)
        self.channel_avg_attention = nn.AdaptiveAvgPool2d(1)  # b,c,1,1
        self.channel_max_attention = nn.AdaptiveMaxPool2d(1)  # b,c,1,1
        self.channel_attention1 = nn.Conv1d(
            in_channels=4, out_channels=1, kernel_size=self.k, padding=self.k // 2)
        self.channel_attention2 = nn.Conv1d(
            in_channels=4, out_channels=1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_attention1 = nn.Conv2d(
            in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.spatial_attention2 = nn.Conv2d(
            in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, encoder, decoder):
        encoder_residual = encoder.clone()
        decoder_residual = decoder.clone()
        encoder_channel_avg_attention = self.channel_avg_attention(
            encoder).squeeze(-1).permute(0, 2, 1)  # b,1,c
        encoder_channel_max_attention = self.channel_max_attention(
            encoder).squeeze(-1).permute(0, 2, 1)  # b,1,c
        decoder_channel_avg_attention = self.channel_avg_attention(
            decoder).squeeze(-1).permute(0, 2, 1)  # b,1,c
        decoder_channel_max_attention = self.channel_max_attention(
            decoder).squeeze(-1).permute(0, 2, 1)  # b,1,c
        encoder_channel_attention = self.channel_attention1(torch.cat(
            [encoder_channel_max_attention, encoder_channel_avg_attention, decoder_channel_max_attention,
             decoder_channel_avg_attention], dim=1)).permute(0, 2, 1).unsqueeze(-1)  # b,c,1,1
        decoder_channel_attention = self.channel_attention2(torch.cat(
            [encoder_channel_max_attention, encoder_channel_avg_attention, decoder_channel_max_attention,
             decoder_channel_avg_attention], dim=1)).permute(0, 2, 1).unsqueeze(-1)  # b,c,1,1
        # encoder = nn.sigmoid(encoder_channel_attention) * encoder
        # decoder = nn.sigmoid(decoder_channel_attention) * decoder
        encoder_spatial_avg_attention = torch.mean(
            encoder, dim=1, keepdim=True)  # b,1,h,w
        encoder_spatial_max_attention = torch.max(
            encoder, dim=1, keepdim=True)[0]  # b,1,h,w
        decoder_spatial_avg_attention = torch.mean(
            decoder, dim=1, keepdim=True)  # b,1,h,w
        decoder_spatial_max_attention = torch.max(
            decoder, dim=1, keepdim=True)[0]  # b,1,h,w
        encoder_spatial_attention = self.spatial_attention1(torch.cat(
            [encoder_spatial_max_attention, encoder_spatial_avg_attention, decoder_spatial_max_attention,
             decoder_spatial_avg_attention], dim=1))  # b,1,h,w
        decoder_spatial_attention = self.spatial_attention2(torch.cat(
            [encoder_spatial_max_attention, encoder_spatial_avg_attention, decoder_spatial_max_attention,
             decoder_spatial_avg_attention], dim=1))  # b,1,h,w
        channel_attention = torch.stack(
            [encoder_channel_attention, decoder_channel_attention], dim=0)  # 2,b,c,1,1
        channel_attention = self.softmax(channel_attention)
        spatial_attention = torch.stack(
            [encoder_spatial_attention, decoder_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_attention = self.softmax(spatial_attention)
        encoder = channel_attention[0] * encoder + \
            spatial_attention[0] * encoder + encoder_residual
        decoder = channel_attention[1] * decoder + \
            spatial_attention[1] * decoder + decoder_residual
        fusion = encoder + decoder
        return fusion


class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_Block, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      stride=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.exchange_channel = Exchange_Encoder_Decoder_Channel()
        self.skip_fusion_enhance = Skip_Fusion_Enhance(out_channels)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoder, decoder):
        decoder = self.upsample(decoder)
        decoder = self.change_channel(decoder)
        encoder, decoder = self.exchange_channel(encoder, decoder)
        fusion = self.skip_fusion_enhance(encoder, decoder)
        out = self.skip(fusion)
        return out


def split_channel(x):
    batch_size, num_channels, height, weight = x.data.size()  # x.shape
    assert int(
        num_channels) % 4 == 0, "The number of channels must be divisible by 4!"
    x = x.reshape(batch_size * num_channels // 2, 2, height *
                  weight).permute(1, 0, 2)  # 2,b*c/2,h*w
    x = x.reshape(2, batch_size, num_channels //
                  2, height, weight)  # 2,b,c/2,h,w
    x1, x2 = x[0], x[1]
    return x1, x2


def kernel_size(channel):
    out = int(math.log2(channel) // 2)
    if out % 2 == 0:
        out += 1
    else:
        out = out
    return out
