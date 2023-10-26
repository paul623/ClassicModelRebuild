"""
@Project ：ClassicModelRebuild 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/10/26 21:13 
"""
from CNN.utils import modelHelper
from model import NiN

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = modelHelper.load_data_fashion_mnist(batch_size, resize=224)
modelHelper.trainAndTest(NiN, train_iter, test_iter, num_epochs, lr)
