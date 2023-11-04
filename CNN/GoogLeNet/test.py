"""
@Project ：ClassicModelRebuild 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/11/3 19:18 
"""
from CNN.utils import modelHelper
from model import GoogLeNet

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = modelHelper.load_data_fashion_mnist(batch_size, resize=96)
modelHelper.trainAndTest(GoogLeNet, train_iter, test_iter, num_epochs, lr)