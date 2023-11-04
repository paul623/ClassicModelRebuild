# PyTorch重构经典网络

自己学习备用，主要参考 《动手学深度学习》+ 小土堆（B站）

跟着学习进度随缘更新

# 环境依赖

可能会用到的

```python
pip install tensorboard 
```

tensorboard日志使用方法也比较简单，直接

```shell
tensorboard --logdir=日志地址
```



# CNN.utils.modelHelper

```python
def trainAndTest(model, train_iter, test_iter, epoch, lr, log_dir="../logs", save_model=False):
```

自行指定logdir地址，save_model是每一epoch保存一次，保存在当前路径下面。

# 更新日志
[2023年10月26日] NiN<br>
[2023年11月3日]  AlexNet<br>
[2023年11月4日]  AutoEncoder