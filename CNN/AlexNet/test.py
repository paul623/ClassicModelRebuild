"""
@Project ：ClassicModelRebuild 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/10/23 16:33 
"""
from CNN.utils import modelHelper
from model import AlexNet

lr, num_epochs, batch_size = 0.01, 20, 128
train_iter, test_iter = modelHelper.load_data_fashion_mnist(batch_size, resize=224)
modelHelper.trainAndTest(AlexNet, train_iter, test_iter, num_epochs, lr)

