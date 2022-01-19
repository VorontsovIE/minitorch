import numpy as np
from layers import Sequential, Linear, Bias, ReLu, LeakyReLu, CrossEntropyLoss

class ImageClassificationModel:
    num_classes = 10
    def __init__(self, linear_1: Linear, bias_1: Bias, linear_2: Linear, bias_2: Bias):
        self._linear_1 = linear_1
        self._bias_1   = bias_1
        self._linear_2 = linear_2
        self._bias_2   = bias_2
        self._model = Sequential([
            self._linear_1, self._bias_1, LeakyReLu(0.1),
            self._linear_2, self._bias_2, LeakyReLu(0.1),
        ])

    @property
    def model(self):
        return self._model

    def forward(self, vec: np.ndarray) -> np.ndarray:
        return self.model.forward(vec)

    def backward(self, vec: np.ndarray):
        return self.model.backward(vec)

    def backward_regularization(self, rate:float=1):
        return self.model.backward_regularization(rate)

    def regularization_loss(self) -> float:
        return self.model.regularization_loss()

    def zero_grad(self):
        return self.model.zero_grad()

    def step_grad(self, rate: float):
        return self.model.step_grad(rate)
    
    def store(self, filename):
        np.savez_compressed(filename,
            linear_1=self._linear_1.weights,
            linear_2=self._linear_2.weights,
            bias_1=self._bias_1.weights,
            bias_2=self._bias_2.weights,
        )

    @classmethod
    def load(cls, filename):
        with np.load(filename) as data:
            linear_1 = data['linear_1']
            linear_2 = data['linear_2']
            bias_1 = data['bias_1']
            bias_2 = data['bias_2']
        input_size = linear_1.shape[1]
        hidden_size = linear_1.shape[0]
        l2_coef = 0.01 / (input_size + hidden_size + cls.num_classes)
        return cls(
            Linear(linear_1, l2_coef=l2_coef),
            Bias(bias_1, l2_coef=l2_coef),
            Linear(linear_2, l2_coef=l2_coef),
            Bias(bias_2, l2_coef=l2_coef),
        )

    @classmethod
    def initialize_random(cls, input_size=28*28, hidden_size=300):
        l2_coef = 0.01 / (input_size + hidden_size + cls.num_classes)
        linear_1  = Linear(np.random.normal(size=(hidden_size, input_size)), l2_coef=l2_coef)
        bias_1 = Bias(np.random.normal(size=(hidden_size, 1)), l2_coef=l2_coef)
        linear_2  = Linear(np.random.normal(size=(cls.num_classes, hidden_size)), l2_coef=l2_coef)
        bias_2 = Bias(np.random.normal(size=(cls.num_classes, 1)), l2_coef=l2_coef)
        return cls(linear_1, bias_1, linear_2, bias_2)
