from CNN.utils import modelHelper
from CNN.GoogLeNet.model import GoogLeNet
from d2l import torch as d2l

lr, num_epochs, batch_size = 0.1, 20, 128
train_iter, test_iter = modelHelper.load_data_fashion_mnist(batch_size, resize=96)

# modelHelper.trainAndTest(GoogLeNet, train_iter, test_iter, num_epochs, lr)
d2l.train_ch6(GoogLeNet(), train_iter, test_iter, num_epochs, lr, d2l.try_gpu())