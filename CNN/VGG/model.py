"""
@Project ：ClassicModelRebuild 
@File    ：model.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/10/23 12:20 
"""
import torch
from torch import nn


# 一个VGG块
# （卷积层的数量，输入通道的数量，输出通道的数量）
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


"""
VGG网络可以分成两个部分
1. 卷积层和汇聚层
2. 全连接层
"""
# 指定每一个块的(卷积层个数， 输出通道数)
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


# 该函数用来搭建VGG的第一部分即卷积层和池化层
def getModelPart1(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return conv_blks


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.model = nn.Sequential(
            *getModelPart1(conv_arch),  # 星号是拆解成一个个的
            nn.Flatten(),
            # 第二部分，全连接部分，和AlexNet保持一致
            nn.Linear(conv_arch[-1][1] * 7 * 7, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# vgg11 = VGG11()
# X = torch.randn(size=(1, 1, 224, 224))
# for blk in vgg11.model:
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)