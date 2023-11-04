"""
@Project ：ClassicModelRebuild 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/11/4 14:51 
"""
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

N_TEST_IMG = 5
epoch, batch_size = 10, 64
lr = 0.005

train_data = torchvision.datasets.MNIST(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)

# plot one example
# 训练数据
print(train_data.train_data.size())     # (60000, 28, 28)
# 训练标签
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[2])
plt.show()

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
loss_func = nn.MSELoss()


# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

# 训练
for epoch in range(epoch):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 28 * 28)  # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        # 比对解码出来的数据和原始数据，计算loss
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 300 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
                plt.draw()
                plt.pause(0.05)
plt.ioff()
plt.show()


# visualize in 3D plot
view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()
