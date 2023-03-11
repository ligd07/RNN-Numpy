class Layers:
    """
        神经网络层模板
        神经网络需要实现以下基本功能函数:
            forward:神经网络前向计算
            backward:神经网络误差反向传播，计算误差梯度
            update:更新神经网络参数
            get_param:获取网络参数
            set_param:设置网络参数
    """
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        pass

    def backward(self, grad_in):
        pass

    def update(self, lr=1e-3):
        pass

    def get_param(self):
        return {}

    def set_param(self, param):
        pass


class Module:
    """
        神经网络模板
        神经网络需要实现以下基本功能函数:
            forward:神经网络前向计算
            backward:神经网络误差反向传播，计算每层误差梯度
            step:更新神经网络所有参数
    """
    def __init__(self):
        self._layers = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if isinstance(value, Layers):
            self._layers.append(value)

    def forward(self, *args, **kwargs):
        """
            依据神经网络的结构与输入计算输出
        """
        pass

    def backward(self, grad):
        """
            误差反向传播计算

        Parameters
        ----------
        grad:numpy.ndarray
            误差梯度
        """
        for layer in reversed(self._layers):
            grad = layer.backward(grad)

    def step(self, lr=1e-3):
        """
            神经网络参数更新

        Parameters
        ----------
        lr:float
            参数更新步长
        """
        for layer in reversed(self._layers):
            layer.update(lr)

    def get_param(self):
        """
            获取神经网络所有参数，并以字典的形式返回

        Returns
        -------
        dic:dict
            神经网络所有参数
        """
        dic = {}
        for layer in self._layers:
            dic.update(layer.get_param())
        return dic

    def set_param(self, param):
        """
            设置神经网络参数

        Parameters
        ----------
        param:dict
            需要设置神经网络参数字典
        """
        # 设置每层神经网络参数
        for layer in self._layers:
            layer.set_param(param)
