import numpy as np
from module import Layers


class RNN(Layers):
    """
        z_t=x_t @ U + h_(t-1) @ W + B

        h_t = tanh(z_t)

        输入为x=[x_1,
                x_2,
                x_3
                .
                .],形状为input_length*input_size
        输出为h=[h_1,
                h_2,
                h_3
                .
                .],形状为input_length*hidden_size
    """
    def __init__(self, name, input_size, hidden_size):
        """
            循环神经网络初始化

        Parameters
        ----------
        name:str
            循环神经网络名称
        input_size:int
            输入向量维度，RNN输入数据是形状为input_length*input_size的二维数组
        hidden_size:int
            隐藏层向量维度，RNN输出数据是形状为input_length*hidden_size的二维数组
        """
        super(RNN, self).__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.U = np.random.randn(self.input_size, self.hidden_size)
        self.W = np.random.randn(self.hidden_size, self.hidden_size)
        self.B = np.random.randn(self.hidden_size)

        self.h_list = np.array([0])
        self.x_list = np.array([0])

        self.grad_U = np.zeros((self.input_size, self.hidden_size))
        self.grad_W = np.zeros((self.hidden_size, self.hidden_size))
        self.grad_B = np.zeros((self.hidden_size,))

    def forward(self, x, h):
        """
            RNN神经网络前向计算公式
        Parameters
        ----------
        x:numpy.ndarray
            输入数据，二维矩阵，大小为input_length*input_size
        h:numpy.ndarray
            输入隐藏层数据，二维矩阵，大小为1*hidden_size

        Returns
        -------
        h_list:numpy.ndarray
            输出数据，二维矩阵，大小为input_length*output_size
        h:numpy.ndarray
            输出隐藏层数据，二维矩阵，大小为1*hidden_size
        """
        self.x_list = x
        self.h_list = np.zeros((self.x_list.shape[0], self.hidden_size))
        for i in range(self.x_list.shape[0]):
            z = self.x_list[i] @ self.U + h @ self.W + self.B
            h = np.tanh(z)
            self.h_list[i] = h

        return self.h_list, h

    def backward(self, grad_in):
        """
            RNN神经网络误差反向传播计算

        Parameters
        ----------
        grad_in:numpy.ndarray
            输入误差梯度，二维矩阵，大小为input_length*hidden_size

        Returns
        -------
        grad_out:numpy.ndarray
            输出误差梯度，二维矩阵，大小为input_length*input_size
        """
        grad_out = np.zeros_like(self.x_list)
        h_buf = (1 - self.h_list ** 2)
        grad_z = grad_in[-1:] * h_buf[-1:]
        grad_out[-1:] = grad_z @ self.U.T

        self.grad_U = self.x_list[-1:].T @ grad_z
        self.grad_W = self.h_list[-2:-1].T @ grad_z
        self.grad_B = grad_z[0]

        for t in range(self.h_list.shape[0] - 2, 0, -1):
            grad_z = (grad_in[t:t + 1] + (grad_z @ self.W.T)) * h_buf[t:t + 1]
            grad_out[t:t + 1] = grad_z @ self.U.T
            self.grad_U += self.x_list[t:t + 1].T @ grad_z
            self.grad_W += self.h_list[t - 1:t].T @ grad_z
            self.grad_B += grad_z[0]

        grad_z = (grad_in[0:1] + (grad_z @ self.W.T)) * h_buf[0:1]
        grad_out[0:1] = grad_z @ self.U.T
        self.grad_U += self.x_list[0:1].T @ grad_z
        self.grad_W += 0
        self.grad_B += grad_z[0]

        for i in [self.grad_U, self.grad_W, self.grad_B]:
            np.clip(i, -5, 5, out=i)

        return grad_out

    def update(self, lr=1e-3):
        """
            参数更新

        Parameters
        ----------
        lr:float
            参数更新步长
        """
        self.U -= lr * self.grad_U
        self.W -= lr * self.grad_W
        self.B -= lr * self.grad_B

    def get_param(self):
        param = {self.name + '-U': self.U, self.name + '-W': self.W, self.name + '-B': self.B}
        return param

    def set_param(self, param):
        # 获取神经网络所有参数，避免传入字典的键值缺失
        d = self.get_param()
        # 更新参数字典
        d.update(param)
        self.U = d[self.name + '-U']
        self.W = d[self.name + '-W']
        self.B = d[self.name + '-B']
