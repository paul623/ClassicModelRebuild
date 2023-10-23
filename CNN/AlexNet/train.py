"""
@Project ：ClassicModelRebuild 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/10/23 11:45 
"""

import torch
from torch.utils.data import DataLoader
import torchvision.datasets
from model import AlexNet
from torch import nn
import time

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.FashionMNIST("../dataset", train=False, transform=transform, download=True)
test_data = torchvision.datasets.FashionMNIST("../dataset", train=True, transform=transform, download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

train_iter = DataLoader(train_data, batch_size=128, drop_last=True)
test_iter = DataLoader(test_data, batch_size=128, drop_last=True)

lr, num_epochs = 0.01, 10
device = torch.device("cuda")

net = AlexNet()
net = net.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)

# 设置训练网络的一些参数
# 训练训练的次数
total_train_step = 0
# 记录训练的次数
total_test_step = 0
# 训练的轮数
epoch = 10

start_time = time.time()
for i in range(epoch):
    print("------------------第{}轮训练------------------".format(i + 1))
    net.train()
    # 训练步骤开始
    for data in train_iter:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        # 使用优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录训练次数
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("用时：{}".format(end_time - start_time))
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
    # 测试步骤
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_iter:
            imgs, targets = data
            imgs = imgs.to(device)  # GPU加速
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss : {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    total_test_step = total_test_step + 1
