import functools
import math
import numpy as np
from typing import Tuple, List

class Layer:
    def forward(self, vec):
        return vec
    def backward(self, vec):
        return vec
    def backward_regularization(self, rate=1):
        pass
    def regularization_loss(self):
        return 0
    def zero_grad(self):
        pass
    def step_grad(self, rate):
        pass

class Linear(Layer):
    def __init__(self, weights, l2_coef=1.0):
        self.weights = weights
        self.grad = np.zeros_like(self.weights)
        self.l2_coef = l2_coef
    def forward(self, vec):
        self.input = vec
        return self.weights @ vec
    def backward(self, vec):
        self.grad += vec @ np.transpose(self.input) # gradient by weights
        return np.transpose(self.weights) @ vec     # gradient by input
    def backward_regularization(self, rate=1):
        self.grad += 2 * rate * self.l2_coef * self.weights
    def regularization_loss(self):
        return self.l2_coef * np.sum(self.weights ** 2)
    def zero_grad(self):
        self.grad = np.zeros_like(self.weights)
    def step_grad(self, rate):
        self.weights -= rate * self.grad

class Bias(Layer):
    def __init__(self, bias, l2_coef=1.0):
        self.bias = bias
        self.grad = np.zeros_like(self.bias)
        self.l2_coef = l2_coef
    def forward(self, vec):
        self.input = vec
        return self.bias + vec
    def backward(self, vec):
        self.grad += vec
        return vec
    def backward_regularization(self, rate=1):
        self.grad += 2 * rate * self.l2_coef * self.bias
    def regularization_loss(self):
        return self.l2_coef * np.sum(self.bias ** 2)
    def zero_grad(self):
        self.grad = np.zeros_like(self.bias)
    def step_grad(self, rate):
        self.bias -= rate * self.grad

class ReLu(Layer):
    def forward(self, vec):
        self.input = vec
        return (vec > 0) * vec
    def backward(self, vec):
        return (self.input > 0) * vec

class LeakyReLu(Layer):
    def __init__(self, alpha):
        self.alpha = alpha
    def forward(self, vec):
        self.input = vec
        return np.where(vec > 0, vec, vec * self.alpha)
    def backward(self, vec):
        return np.where(self.input > 0, 1, self.alpha) * vec

class Sequential(Layer):
    def __init__(self, layers):
        self.layers = layers
    def forward(self, vec):
        return functools.reduce(lambda v, layer: layer.forward(v), self.layers, vec)
    def backward(self, vec):
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
    def forward(self, vec):
        self.input = vec
        exps = np.exp(vec)
        return exps / np.sum(exps)
    def backward(self, vec):
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
    def forward(self, vec, label):
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
    #
    def backward(self):
        # dH/dx_i = \exp{v_i} / (\sum_k\exp{v_k}) - \delta_{i,l} =
        #         = \exp{v_i - m} / (\sum_k\exp{v_k - m}) - \delta_{i,l}
        #         = \exp{w_i} / (\sum_k\exp{w_k}) - \delta_{i,l}
        result = self.shifted_exp / self.exp_sum
        result[self.label] -= 1
        return result
