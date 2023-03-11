import numpy as np
import matplotlib.pyplot as plt
from module import Module
from rnn import RNN
from bp import BP
from activate import SoftMax

hidden_size = 4
np.random.seed(0)


# 定义误差函数
def loss(y_hat, y_true):
    a = y_true * np.log(y_hat)
    return -a.sum(), -y_true / y_hat


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = RNN(name="RNN", input_size=4, hidden_size=hidden_size)
        self.bp1 = BP(name="BP", input_size=hidden_size, output_size=4)
        self.softmax = SoftMax(name="softmax", axis=1)

    def forward(self, x, h):
        y, h = self.rnn(x, h)
        y = self.bp1(y)
        y = self.softmax(y)
        return y, h


# 训练函数
def train(x, y):
    epochs = 5000
    lr = 0.1
    e = []
    for i in range(epochs):
        y_hat, h = nn(x, np.zeros((hidden_size, )))
        l, grad = loss(y_hat, y)
        e.append(l)
        # 反向传播求梯度
        nn.backward(grad)
        # 跟新梯度
        nn.step(lr)
    return e


# 构造训练样本
#  欢 迎 光 临
# 欢->   [1,0,0,0]
# 迎->   [0,1,0,0]
# 光-> [0,0,1,0]
# 临!->   [0,0,0,1]
data = ['欢', '迎', '光', '临!\n']
X = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
Y = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
])

if __name__ == '__main__':
    # 实例化神经网络
    nn = Net()
    # 读取参数
    # d = np.load('./parameters/sen_param.npz')
    # nn.set_param(d)
    # 训练神经网络
    e = train(X, Y)
    # 保存参数
    # d = nn.get_param()
    # np.savez('./parameters/sen_param.npz', **d)
    plt.plot(e)

    # 测试
    h = np.zeros((hidden_size, ))
    # 预测输入：‘人’
    x = np.array([[0, 0, 0, 1]])
    y_list = np.zeros((16, 4))
    for i in range(y_list.shape[0]):
        y, h = nn(x, h)
        x = y
        y_list[i] = y

    for i in y_list.argmax(axis=1):
        print(data[i], end='')

    plt.show()
