"""
@Project ：ClassicModelRebuild
@File    ：model.py
@IDE     ：PyCharm
@Author  ：paul623
@Date    ：2023/10/23 11:21
根据李沐 动手学深度学习
"""

from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减少卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用3个连续的卷积层和最小的卷积窗口
            # 除了最后的卷积层，输出通道数进一步增加
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 这个操作通常用在卷积神经网络（CNN）的卷积层之后，以将卷积层的输出张量展平为一维向量
            # 然后将其传递给全连接层（通常是一个或多个线性层），以进行分类或其他任务。
            # 展平操作有助于将卷积层提取的特征转化成适合全连接层处理的形式。
            nn.Flatten(),
            # 这里，全连接层输出数量是LeNet中的好几倍。使用暂退层来缓解过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            # 在全连接层（Linear 层）之间添加 Dropout 层是一种常见的做法
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# X = torch.randn(1, 1, 224, 224)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)
