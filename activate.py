import numpy as np
from module import Layers


class Sigmoid(Layers):
    def __init__(self, name):
        """
            激活层初始化

        Parameters
        ----------
        name:str
            激活层名称
        """
        super(Sigmoid, self).__init__(name)
        self.data_out = None

    def forward(self, data_in):
        """
            激活层前向计算

        Parameters
        ----------
        data_in:float|int|numpy.ndarray
            激活层输入
        Returns
        -------
        data_out:float|int|numpy.ndarray
            激活层输出
        """
        self.data_out = 1 / (1 + np.exp(-data_in))
        return self.data_out

    def backward(self, grad_in):
        """
            激活层误差反向传播计算

        Parameters
        ----------
        grad_in:float|int|numpy.ndarray
            误差梯度输入
        Returns
        -------
        grad_out:float|int|numpy.ndarray
            误差梯度输出
        """
        grad_in = grad_in.reshape(self.data_out.shape)
        grad_out = grad_in * self.data_out * (1 - self.data_out)
        return grad_out


class SoftMax(Layers):
    def __init__(self, name, axis=1):
        """
            softmax层初始化
        Parameters
        ----------
        name:str
            softmax层名称
        axis:int
            指定计算轴向
        """
        super(SoftMax, self).__init__(name)
        self.axis = axis
        self.data_output = None

    def forward(self, data_in):
        """
            softmax层前向计算

        Parameters
        ----------
        data_in:numpy.ndarray
            softmax层输入
        Returns
        -------
        data_output:numpy.ndarray
            softmax层输出
        """
        result = np.exp(data_in)
        self.data_output = result / np.expand_dims(result.sum(self.axis), self.axis)
        return self.data_output

    def backward(self, grad_in):
        """
            softmax层误差反向传播计算

        Parameters
        ----------
        grad_in:numpy.ndarray
            误差梯度输入
        Returns
        -------
        grad_out:numpy.ndarray
            误差梯度输出
        """
        grad_out = grad_in * self.data_output - \
                   np.expand_dims((grad_in * self.data_output).sum(self.axis), self.axis) * \
                   self.data_output
        return grad_out
