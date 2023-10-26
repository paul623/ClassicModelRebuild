"""
@Project ：ClassicModelRebuild 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/10/23 16:15 
"""
from CNN.utils import modelHelper
from model import VGG11

lr, num_epochs, batch_size = 0.05, 10, 32
train_iter, test_iter = modelHelper.load_data_fashion_mnist(batch_size, resize=224)
modelHelper.trainAndTest(VGG11, train_iter, test_iter, num_epochs, lr)