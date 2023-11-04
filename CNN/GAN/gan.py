"""
@Project ：ClassicModelRebuild 
@File    ：gan.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/10/31 11:12 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import pandas

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 对数据做归一化（-1 --- 1）
tranform = transforms.Compose([
    transforms.ToTensor(),  # 0-1归一化：channel,high,width
    transforms.Normalize(0.5, 0.5),  # 均值，方差均为0.5
])

# 训练数据集
train_ds = torchvision.datasets.MNIST("data", train=True, transform=tranform, download=True)

dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

imgs, _ = next(iter(dataloader))
print(imgs.shape)  # torch.Size([64, 1, 28, 28])


# batch_size=64 图片大小：1,28,28
# 定义生成器
# 输入是长度为100的噪声（正态分布随机数）
# 输出为（1,28,28）的图片

# linear1:100---256
# linear2:256---512
# linear1:512---28*28
# linear2:28*28---(1,28,28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):  # x表示长度为100的噪声输入
        img = self.main(x)
        # view()相当于reshape、resize，重新调整Tensor的形状。
        # view()函数常用于调整输入数据的形状以适应模型的输入要求，或者在网络层之间传递数据时进行形状的转换
        img = img.view(-1, 28, 28)
        return img


# 定义判别器
# 输入为（1,28,28）的图片，输出为二分类的概率值，输出使用sigmoid激活
# BCELose计算交叉熵损失
# nn.LeakyReLU  f(x):x>0,输出x，如果x<0,输出a*x，a表示一个很小的斜率，比如0.1
# 判别器中一般推荐使用nn.LeakyReLU
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.main(x)
        return x


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if 0 else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)

loss_fn = torch.nn.BCELoss()


def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    # detach()截断梯度,np.squeeze可以去掉维度为1
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):  # prediction.size(0)=16
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2)  # tanh 得到的是-1 - 1之间，-》0-1之间
        plt.axis("off")
    plt.show()


test_input = torch.randn(16, 100, device=device)  # 16个长度为100的正态随机数

# print(test_input)

D_loss = []
G_loss = []

# 训练循环
for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)  # len(dataloader)返回批次数
    # len(dataset)返回样本数
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)

        d_optim.zero_grad()
        real_output = dis(img)  # 对判别器输入真实图片 real_output对真实图片的预测结果
        # 得到判别器在真实图像上面的损失
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))

        d_real_loss.backward()

        gen_img = gen(random_noise)
        # detach()截断生成器梯度，更新判别器梯度
        fake_output = dis(gen_img.detach())  # 判别器输入生成图片。fake_output对生成图片的预测
        # 得到判别器在生成图像上面的损失
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))

        d_fake_loss.backward()

        d_loss = d_fake_loss + d_real_loss
        d_optim.step()

        g_optim.zero_grad()
        fake_output = dis(gen_img)
        # 得到生成器的损失
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output))

        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print("Epoch", epoch)

        gen_img_plot(gen, test_input)