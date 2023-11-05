""" 基于MNIST 实现条件对抗生成网络 (CGAN) """
import os

import torch
import torchvision
import torch.nn as nn
import numpy as np

image_size = [1, 28, 28]
latent_dim = 96
label_emb_dim = 32
batch_size = 64
use_gpu = torch.cuda.is_available()
save_dir = "cgan_images"
os.makedirs(save_dir, exist_ok=True)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        '''
        Embedding(10, 32) 长度为10的张量，每个张量的大小是32
        10个手写体类别
        https://zhuanlan.zhihu.com/p/647536930
        '''
        self.embedding = nn.Embedding(10, label_emb_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim+label_emb_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),

            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),

            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),

            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),

            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),  # prod计算数组中所有元素的乘积。 其实等价于1*28*28 花里胡哨的
            #  nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z, labels):
        # shape of z: [batchsize, latent_dim]
        label_embedding = self.embedding(labels)
        z = torch.cat([z, label_embedding], dim=-1)

        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)

        return image


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.embedding = nn.Embedding(10, label_emb_dim)
        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size, dtype=np.int32) + label_emb_dim, 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, labels):
        # shape of image: [batchsize, 1, 28, 28]

        label_embedding = self.embedding(labels)
        prob = self.model(torch.cat([image.reshape(image.shape[0], -1), label_embedding], dim=-1))

        return prob


# Training
dataset = torchvision.datasets.MNIST("../dataset", train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                             #  torchvision.transforms.Normalize([0.5], [0.5]),
                                         ]
                                     )
                                     )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

generator = Generator()
discriminator = Discriminator()

'''
这是Adam优化器的betas参数，它控制了一阶矩估计（mean）和二阶矩估计（uncentered variance）的衰减率。
具体而言，(0.4, 0.8) 表示使用较小的衰减率来估计一阶和二阶矩。这有助于稳定训练和防止梯度爆炸。
通常，这些值的默认设置（0.9和0.999）也可以工作良好，但可以根据需要进行调整
weight_decay=0.0001：这是权重衰减（weight decay）的超参数。
它是L2正则化的一部分，用于防止模型过拟合。设置一个较小的权重衰减可以帮助控制模型参数的大小，以减小过拟合的风险。
'''
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)

if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

num_epoch = 200
for epoch in range(num_epoch):
    print(f"---------------------epoch:{epoch}----------------------")
    for i, mini_batch in enumerate(dataloader):
        gt_images, labels = mini_batch

        z = torch.randn(batch_size, latent_dim)

        if use_gpu:
            gt_images = gt_images.to("cuda")
            z = z.to("cuda")
            labels = labels.to("cuda")

        pred_images = generator(z, labels)
        g_optimizer.zero_grad()

        recons_loss = torch.abs(pred_images - gt_images).mean()

        g_loss = recons_loss * 0.05 + loss_fn(discriminator(pred_images, labels), labels_one)

        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()

        real_loss = loss_fn(discriminator(gt_images, labels), labels_one)
        fake_loss = loss_fn(discriminator(pred_images.detach(), labels), labels_zero)
        d_loss = (real_loss + fake_loss)

        # 观察real_loss与fake_loss，同时下降同时达到最小值，并且差不多大，说明D已经稳定了

        d_loss.backward()
        d_optimizer.step()

        if i % 300 == 0:
            print(
                f"step:{len(dataloader) * epoch + i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        if i % 800 == 0:
            image = pred_images[:16].data
            torchvision.utils.save_image(image, f"image_{len(dataloader) * epoch + i}.png", nrow=4)
