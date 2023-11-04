"""
@Project ：ClassicModelRebuild 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/11/3 19:17 
"""
from CNN.utils import modelHelper
from model import DenseNet

lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = modelHelper.load_data_fashion_mnist(batch_size, resize=96)
modelHelper.trainAndTest(DenseNet, train_iter, test_iter, num_epochs, lr)