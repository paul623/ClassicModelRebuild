"""
@Project ：ClassicModelRebuild 
@File    ：modelHelper.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/10/23 15:52 
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time
import d2l


# 加载FASHION数据集
def load_data_fashion_mnist(batch_size, resize, root="../dataset"):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.ToTensor()
    ])
    train_data = torchvision.datasets.FashionMNIST(root, train=False, transform=transform, download=True)
    test_data = torchvision.datasets.FashionMNIST(root, train=True, transform=transform, download=True)
    train_iter = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    test_iter = DataLoader(test_data, batch_size=batch_size, drop_last=True)
    return train_iter, test_iter



def getAvailableDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# 训练 测试 保存 日志
# 优化器可以自己定义，直接改源码吧~
def trainAndTest(model, train_iter, test_iter, epoch, lr, log_dir="../logs", save_model=False):
    device = getAvailableDevice()
    train_data_size = len(train_iter.dataset)
    test_data_size = len(test_iter.dataset)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # 创建网络
    net = model()
    net.to(device)  # 设置在指定设备上运行

    net.apply(init_weights)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 设置训练网络的一些参数
    # 训练训练的次数
    total_train_step = 0
    # 记录训练的次数
    total_test_step = 0
    # 添加tensorboard
    writer = SummaryWriter(log_dir)
    start_time = time.time()
    for i in range(epoch):
        print("------------------第{}轮训练------------------".format(i + 1))
        net.train()
        # 训练步骤开始
        for data in train_iter:
            optimizer.zero_grad()
            imgs, targets = data
            imgs = imgs.to(device)  # GPU加速
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            # 优化器优化模型
            loss.backward()
            optimizer.step()
            # 记录训练次数
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print("用时：{}".format(end_time - start_time))
                print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        # 测试步骤开始
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
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size)
        total_test_step = total_test_step + 1
        if save_model:
            torch.save(net, "MyModel_{}.pth".format(i))
            print("模型已保存")
    writer.close()

