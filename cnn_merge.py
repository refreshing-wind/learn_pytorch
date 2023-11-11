#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cnn_merge.py
@Time    :   2023/08/11 09:21:07
@Author  :   xuexin 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import time

in_channels = 2
ou_channels = 2
kernel_size =3
w = 9
h = 9
# res_block = 3*3 conv + 1*1 conv + input
x = torch.ones(1, in_channels, w, h) # 输入图片
torch.manual_seed(9)
# 方法1：原生方法
t1 = time.time()
conv_2d = nn.Conv2d(in_channels, ou_channels, kernel_size, padding="same")
# print(conv_2d.weight)
# print(conv_2d.weight.data)
conv_2d_pointwise = nn.Conv2d(in_channels, ou_channels, 1)
# result1 = conv_2d(x) + conv_2d_pointwise(x) + x
# print(result1)
# print(conv_2d_pointwise(x))
# 方法二：算子融合
# 把point-wise卷积和x本身都写成3*3的卷积
# 最终把三个卷积写出一个卷积
pointwise_to_conv_weight = F.pad(conv_2d_pointwise.weight, [1,1,1,1,0,0,0,0]) # 2*2*1*1->2*2*3*3
conv_2d_for_pointwise = nn.Conv2d(in_channels, ou_channels, kernel_size, padding='same')
conv_2d_for_pointwise.weight = nn.Parameter(pointwise_to_conv_weight)
conv_2d_for_pointwise.bias = conv_2d_pointwise.bias  
# 2*2*3*3
zeros = torch.unsqueeze(torch.zeros(kernel_size, kernel_size), 0)
stars = torch.unsqueeze(F.pad(torch.ones(1,1), [1,1,1,1]), 0)
# print(zeros.shape)
stars_zeros = torch.unsqueeze(torch.cat([stars, zeros], 0), 0)
zeros_stars = torch.unsqueeze(torch.cat([zeros, stars], 0), 0)
identity_to_conv_weight = torch.cat([stars_zeros, zeros_stars], 0)
identity_to_conv_bias = torch.zeros([ou_channels])
conv_2d_for_identity = nn.Conv2d(in_channels, ou_channels, kernel_size, padding='same')
conv_2d_for_identity.weight = nn.Parameter(identity_to_conv_weight)
conv_2d_for_identity.bias = nn.Parameter(identity_to_conv_bias)

result2 = conv_2d(x) + conv_2d_for_pointwise(x) + conv_2d_for_identity(x)
# print(conv_2d_for_pointwise(x))
# print(result2)
# print(torch.all(torch.isclose(result1, result2)))

# 2） 融合
conv_2d_for_fusion = nn.Conv2d(in_channels, ou_channels, kernel_size, padding='same')
conv_2d_for_fusion.weight = nn.Parameter(conv_2d.weight.data+conv_2d_for_pointwise.weight.data+conv_2d_for_identity.weight.data)
conv_2d_for_fusion.bias = nn.Parameter(conv_2d.bias.data+conv_2d_for_pointwise.bias.data+conv_2d_for_identity.bias.data)
result3 = conv_2d_for_fusion(x)
print(torch.all(torch.isclose(result2, result3)))
t2 = time.time()

conv_2d = nn.Conv2d(in_channels, ou_channels, kernel_size, padding="same")
# print(conv_2d.weight)
# print(conv_2d.weight.data)
conv_2d_pointwise = nn.Conv2d(in_channels, ou_channels, 1)
result1 = conv_2d(x) + conv_2d_pointwise(x) + x
t3 = time.time()
print(f"原生时间{t3-t2}")
print(f"融合时间{t2-t1}")