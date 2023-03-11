import numpy as np
import matplotlib.pyplot as plt
from module import Module
from rnn import RNN
from bp import BP
import datetime

hidden_size = 14
np.random.seed(0)


# 定义误差函数
def loss(y_hat, y_true):
    data_out = ((y_true - y_hat) ** 2).mean()
    grad_out = 2 * (y_hat - y_true)
    return data_out, grad_out


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn1 = RNN(name="RNN1", input_size=1, hidden_size=hidden_size)
        self.rnn2 = RNN(name="RNN2", input_size=hidden_size, hidden_size=hidden_size)
        self.rnn3 = RNN(name="RNN3", input_size=hidden_size, hidden_size=hidden_size)
        self.bp1 = BP(name="BP", input_size=hidden_size, output_size=1)

    def forward(self, x, h):
        y, h[0] = self.rnn1(x, h[0])
        y, h[1] = self.rnn2(y, h[1])
        y, h[2] = self.rnn3(y, h[2])
        y = self.bp1(y)
        return y, h


# 训练函数
def train():
    # 训练次数
    epochs = 50000
    # 参数更新率
    lr = 0.001
    # 正弦信号开始时刻，在0~1之间随机变化
    start = np.random.uniform(size=epochs)
    # 记录误差
    e = []

    print('\r\n开始训练:', datetime.datetime.now())

    for i in range(epochs):
        # 生成训练数据
        a = np.arange(start[i], start[i]+10, 0.2)
        b = np.sin(a)
        x = b[:-1].reshape(-1, 1)
        y = b[1:].reshape(-1, 1)

        # 前向计算
        y_hat, h = nn(x, np.zeros((3, hidden_size)))
        # 误差计算
        l, grad = loss(y_hat, y)
        # 记录误差
        e.append(l)
        # 反向传播
        nn.backward(grad)
        # 更新梯度
        nn.step(lr)
        # 打印训练进度
        if i % 100 == 0:
            r = round(i / epochs * 100, 2)
            print(f'\r训练已完成练{r}%,误差:{round(l, 6)}', end='')

    print(f'\r训练已完成练100%,误差:{round(l, 6)}')
    print('结束训练:', datetime.datetime.now())

    return e


# 测试函数
def test(start=0.0):
    # 生成正确正弦信号，用来与预测输出进行对比
    a = np.arange(start, start + 15, 0.2)
    b = np.sin(a)
    X = b.reshape(-1, 1)

    # 隐藏层输入，一般为0矩阵
    h = np.zeros((3, hidden_size))
    # 预测的初始值
    x_t = np.zeros((1, 1))
    x_t[0:1] = X[0:1]
    # 预测序列输出
    y = np.zeros((a.shape[0], 1))

    for i in range(a.shape[0]):
        # 预测下一个点的输出
        yhat, h = nn(x_t, h)
        # 将预测值输入RNN网络继续预测下一个值
        x_t = yhat
        # 保存预测值
        y[i] = yhat

    # 绘制预测起始点
    plt.scatter(a[0:1], X[0:1], color='r', label='输入')
    # 绘制预测序列
    plt.scatter(a[1:], y[:-1], color='y', label='预测')
    # 绘制正弦波
    plt.plot(a, X, label='正弦波')
    # 设置标签位置及大小
    plt.legend(loc="lower left", fontsize=10)


if __name__ == '__main__':
    # 实例化神经网络
    nn = Net()
    # 读取参数
    # nn.set_param(np.load('./parameters/sin_param.npz'))
    # 训练神经网络
    e = train()
    # 保存参数
    # np.savez('./parameters/sin_param.npz', **(nn.get_param()))
    plt.plot(e)

    plt.figure(2)
    # 设置标签字体
    plt.rc('font', family='STSong')
    # 绘制起始点为0,0.2,0.4,0.6,0.8,1.0的测试结果
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        test(0.2*i)
        plt.title('start-' + str(round(i*0.2, 1)))
    plt.show()
