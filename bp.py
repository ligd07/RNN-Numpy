import numpy as np
from module import Layers


class BP(Layers):
    def __init__(self, name, input_size, output_size):
        """
            BP神经网络初始化

        Parameters
        ----------
        name:str
            BP神经网络层名称
        input_size:int
            输入向量维度
        output_size:int
            输出向量维度
        """
        super(BP, self).__init__(name)
        self.input_size = input_size
        self.output_size = output_size

        self.W = np.random.randn(self.input_size, self.output_size)
        self.B = np.random.randn(self.output_size)

        self.grad_W = np.zeros((self.input_size, self.output_size))
        self.grad_B = np.zeros((self.output_size,))

        self.data_input = None

    def forward(self, data_in):
        """
            BP神经网络前向计算

        Parameters
        ----------
        data_in:numpy.ndarray
            输入数据，二维矩阵，大小为input_length*in_channels
        Returns
        -------
        data_out:numpy.ndarray
            输出数据，二维矩阵，大小为大小为input_length*out_channels
        """
        self.data_input = data_in
        data_out = self.data_input @ self.W + self.B
        return data_out

    def backward(self, grad_in):
        """
            BP神经网络误差反向传播计算

        Parameters
        ----------
        grad_in:numpy.ndarray
            输入误差梯度，二维矩阵，大小为input_length*out_channels
        Returns
        -------
        grad_out:numpy.ndarray
            输出误差梯度，二维矩阵，大小为input_length*out_channels
        """
        grad_out = grad_in @ self.W.T

        self.grad_W = self.data_input.T @ grad_in
        self.grad_B = np.ones(grad_in.shape[0]) @ grad_in

        return grad_out

    def update(self, lr=1e-3):
        """
            参数更新

        Parameters
        ----------
        lr:float
            参数更新步长
        """
        self.W -= lr * self.grad_W
        self.B -= lr * self.grad_B

    def get_param(self):
        param = {self.name + '-W': self.W, self.name + '-B': self.B}
        return param

    def set_param(self, param):
        # 获取神经网络所有参数，避免传入字典的键值缺失
        d = self.get_param()
        # 更新参数字典
        d.update(param)
        self.W = d[self.name + '-W']
        self.B = d[self.name + '-B']
