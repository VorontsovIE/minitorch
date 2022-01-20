import numpy as np
from layers import Sequential, Linear, Bias, ReLu, LeakyReLu, CrossEntropyLoss, Model
from typing import Dict

class ImageClassificationModel(Model):
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

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
          'linear_1': self._linear_1.weights,
          'linear_2': self._linear_2.weights,
          'bias_1': self._bias_1.weights,
          'bias_2': self._bias_2.weights,
        }

    @classmethod
    def from_parameters(cls, parameters: Dict[str, np.ndarray]) -> 'ImageClassificationModel':
        input_size = parameters['linear_1'].shape[1]
        hidden_size = parameters['linear_1'].shape[0]
        l2_coef = 0.01 / (input_size + hidden_size + cls.num_classes)
        return cls(
            Linear(parameters['linear_1'], l2_coef=l2_coef),
            Bias(parameters['bias_1'], l2_coef=l2_coef),
            Linear(parameters['linear_2'], l2_coef=l2_coef),
            Bias(parameters['bias_2'], l2_coef=l2_coef),
        )

    @classmethod
    def initialize_random(cls, input_size: int = 28*28, hidden_size: int = 300) -> 'ImageClassificationModel':
        l2_coef = 0.01 / (input_size + hidden_size + cls.num_classes)
        linear_1  = Linear(np.random.normal(size=(hidden_size, input_size)), l2_coef=l2_coef)
        bias_1 = Bias(np.random.normal(size=(hidden_size, 1)), l2_coef=l2_coef)
        linear_2  = Linear(np.random.normal(size=(cls.num_classes, hidden_size)), l2_coef=l2_coef)
        bias_2 = Bias(np.random.normal(size=(cls.num_classes, 1)), l2_coef=l2_coef)
        return cls(linear_1, bias_1, linear_2, bias_2)
