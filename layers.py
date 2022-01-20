import functools
import math
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger('minitorch')

class Layer:
    '''
    Layer is any building block which passes vectors and gradients.
    It needn't be a single layer (see `Sequential`).

    ToDo: make it possible to work with batches. Now only the model works only with single samples.
    ToDo: make it possible to work with multidimensional data.
    '''

    def forward(self, vec: np.ndarray) -> np.ndarray:
        '''
        Pass a single sample feature vector through the model.
        It should store the data necessary to pass gradient backward in the future.
        '''
        return vec

    def backward(self, vec: np.ndarray) -> np.ndarray:
        '''
        Pass loss of a single sample through the model.
        Transforms gradient vector with regards to layer output into gradient with regards to layer input.
        Backward pass should be performed immediately after forward pass.
        More precisely, state stored by forward pass shouldn't be corrupted at the time backward.

        Note (and ToDo)! This interface probably will be replaced when AST is implemented.
        That's because binary operations have gradients wrt both arguments.
        '''
        return vec

    def backward_regularization(self, rate: float = 1):
        '''
        Dirty hack to make an L2 model regularizaton, while AST not implemented.
        Updates gradient vector.
        `rate` is supposed to equal to a number of samples in a batch:
          we update gradient vector once per batch
        '''
        pass

    def regularization_loss(self) -> float:
        "Dirty hack to make a model regularizaton while AST not implemented. Loss value."
        return 0

    def zero_grad(self):
        "Nullify gradients."
        pass

    def step_grad(self, rate: float):
        "Update weights in the anti-gradient direction."
        pass


class Model(Layer):
    '''
    Add load/store functionality to a Layer.
    It's necessary to implement `model`, `parameters` and `cls.from_parameters` methods.
    '''
    @property
    def model(self):
        "Initialize model in constructor and return it in this method."
        raise NotImplementedError()

    def parameters(self) -> Dict[str, np.ndarray]:
        '''
        Return a mapping {parameter_name: parameter_value} 
          where parameter_value usually represent layer weights.
        '''

        raise NotImplementedError()

    @classmethod
    def from_parameters(cls, parameters: Dict[str, np.ndarray]) -> 'Model':
        "Create a model from a mapping of parameters"
        raise NotImplementedError()

    @classmethod
    def load(cls, filename: str) -> 'Model':
        with np.load(filename) as data:
            return cls.from_parameters(data)

    def store(self, filename: str):
        np.savez_compressed(filename, **self.parameters())

    def forward(self, vec: np.ndarray) -> np.ndarray:
        return self.model.forward(vec)

    def backward(self, vec: np.ndarray) -> np.ndarray:
        return self.model.backward(vec)

    def backward_regularization(self, rate: float = 1):
        return self.model.backward_regularization(rate)

    def regularization_loss(self) -> float:
        return self.model.regularization_loss()

    def zero_grad(self):
        return self.model.zero_grad()

    def step_grad(self, rate: float):
        return self.model.step_grad(rate)


class Linear(Layer):
    def __init__(self, weights: np.ndarray, l2_coef: float = 1.0):
        self.weights = weights
        self.grad = np.zeros_like(self.weights)
        self.l2_coef = l2_coef

    def forward(self, vec: np.ndarray) -> np.ndarray:
        self.input = vec
        return self.weights @ vec

    def backward(self, vec: np.ndarray) -> np.ndarray:
        self.grad += vec @ np.transpose(self.input) # gradient by weights
        return np.transpose(self.weights) @ vec     # gradient by input

    def backward_regularization(self, rate: float = 1):
        self.grad += 2 * rate * self.l2_coef * self.weights

    def regularization_loss(self) -> float:
        # Loss = Loss_{pred} + l2_coef * ||W||_{L2}^2
        return self.l2_coef * np.sum(self.weights ** 2)

    def zero_grad(self):
        self.grad = np.zeros_like(self.weights)

    def step_grad(self, rate: float):
        if np.any(np.isnan(self.grad) | np.isinf(self.grad)):
            logger.warning("Weights update failed: NaN/Inf gradients. Skip the batch.")
            return
        self.weights -= rate * self.grad

class Bias(Layer):
    def __init__(self, weights: np.ndarray, l2_coef: float = 1.0):
        self.weights = weights
        self.grad = np.zeros_like(self.weights)
        self.l2_coef = l2_coef

    def forward(self, vec: np.ndarray) -> np.ndarray:
        self.input = vec
        return self.weights + vec

    def backward(self, vec: np.ndarray) -> np.ndarray:
        self.grad += vec
        return vec

    def backward_regularization(self, rate: float = 1):
        self.grad += 2 * rate * self.l2_coef * self.weights

    def regularization_loss(self) -> float:
        # Loss = Loss_{pred} + l2_coef * ||W||_{L2}^2
        return self.l2_coef * np.sum(self.weights ** 2)

    def zero_grad(self):
        self.grad = np.zeros_like(self.weights)

    def step_grad(self, rate: float):
        if np.any(np.isnan(self.grad) | np.isinf(self.grad)):
            logger.warning("Weights update failed: NaN/Inf gradients. Skip the batch.")
            return
        self.weights -= rate * self.grad

class ReLu(Layer):
    def forward(self, vec: np.ndarray) -> np.ndarray:
        self.input = vec
        return (vec > 0) * vec

    def backward(self, vec: np.ndarray) -> np.ndarray:
        return (self.input > 0) * vec

class LeakyReLu(Layer):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, vec: np.ndarray) -> np.ndarray:
        self.input = vec
        return np.where(vec > 0, vec, vec * self.alpha)

    def backward(self, vec: np.ndarray) -> np.ndarray:
        return np.where(self.input > 0, 1, self.alpha) * vec

class Sequential(Layer):
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, vec: np.ndarray) -> np.ndarray:
        return functools.reduce(lambda v, layer: layer.forward(v), self.layers, vec)

    def backward(self, vec: np.ndarray) -> np.ndarray:
        return functools.reduce(lambda v, layer: layer.backward(v), reversed(self.layers), vec)

    def backward_regularization(self, rate=1):
        for layer in self.layers:
            layer.backward_regularization(rate=rate)

    def regularization_loss(self):
        return sum(layer.regularization_loss() for layer in self.layers)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def step_grad(self, rate):
        for layer in self.layers:
            layer.step_grad(rate)


class SoftMax(Layer):
    def forward(self, vec: np.ndarray) -> np.ndarray:
        self.input = vec
        exps = np.exp(vec)
        return exps / np.sum(exps)

    def backward(self, vec: np.ndarray) -> np.ndarray:
        exps = np.exp(self.input).squeeze()
        normed_exps = exps / np.sum(exps)
        jacobian = np.diag(normed_exps) - normed_exps @ np.transpose(normed_exps)
        return jacobian @ vec


class CrossEntropyLoss:
    '''
    Cross entropy of soft-maxed input against a certain class label.
    NOTE! Input vector is raw values, not probabilites vector.
          Softmax of input is calculated inside of CE-loss.
    '''
    def forward(self, vec: np.ndarray, label: int) -> float:
        # H = -p*log(q) where p is actual class distribution, q is predicted distribution
        #   = -1*log(q_l) = -log(SoftMax_l(v_1..v_n))  where l is label
        #   = -log(\exp{v_l} / (\sum_k \exp{v_k}) )
        #   = log(\sum_k \exp{v_k}) - v_l
        #   = log(\exp{m}\sum_k \exp{v_k - m}) - v_l   for any m
        #   = log(\sum_k \exp{v_k - m}) + m - v_l      we put m = max(v_k)
        #   = log(\sum_k \exp{w_k}) - w_l              where w_i = v_i - m
        self.label = label
        shifted_vec = vec - np.max(vec)
        shifted_exp = np.exp(shifted_vec)
        exp_sum = np.sum(shifted_exp)
        self.shifted_vec = shifted_vec
        self.shifted_exp = shifted_exp
        self.exp_sum = exp_sum
        return math.log(exp_sum) - shifted_vec[label]

    def backward(self) -> np.ndarray:
        # dH/dx_i = \exp{v_i} / (\sum_k\exp{v_k}) - \delta_{i,l} =
        #         = \exp{v_i - m} / (\sum_k\exp{v_k - m}) - \delta_{i,l}
        #         = \exp{w_i} / (\sum_k\exp{w_k}) - \delta_{i,l}
        result = self.shifted_exp / self.exp_sum
        result[self.label] -= 1
        return result
