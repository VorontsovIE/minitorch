import functools
import math
import numpy as np
import random
from typing import Tuple, List
from utils import train_test_split, BatchSampler

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

def softmax(vec):
    shifted_vec = vec - np.max(vec)
    shifted_exp = np.exp(shifted_vec)
    exp_sum = np.sum(shifted_exp)
    return shifted_exp / exp_sum


def accuracy(model, data, labels):
    num_samples = data.shape[1]
    correct = 0
    for idx in range(num_samples):
        sample = np.expand_dims(data[:, idx], 1)
        activations = model.forward(sample)
        if predict_label(activations) == labels[idx]:
            correct += 1
    return correct / num_samples


def predict_label(activations):
    return np.argmax(activations.squeeze())

def main():
    seed = 138
    random.seed(seed)
    np.random.seed(seed)

    # pos — label 1
    # neg — label 0
    train_pos = np.random.normal(size=(10,2), loc=(5.0, 1.0), scale=1.0).transpose()
    train_neg = np.random.normal(size=(10,2), loc=(1.0, 5.0), scale=1.0).transpose()
    train_data = np.hstack([train_pos, train_neg])
    train_labels = np.hstack([1 * np.ones(train_pos.shape[1], dtype=int), 0 * np.ones(train_neg.shape[1], dtype=int)])

    test_pos = np.random.normal(size=(10,2), loc=(5.0, 1.0), scale=1.0).transpose()
    test_neg = np.random.normal(size=(10,2), loc=(1.0, 5.0), scale=1.0).transpose()
    test_data = np.hstack([test_pos, test_neg])
    test_labels = np.hstack([1 * np.ones(test_pos.shape[1], dtype=int), 0 * np.ones(test_neg.shape[1], dtype=int)])


    l2_coef = 0.01
    lin_1 = Linear(np.random.normal(size=(3,2)), l2_coef = l2_coef)
    bias_1 = Bias(np.random.normal(size=(3,1)), l2_coef = l2_coef)
    lin_2 = Linear(np.random.normal(size=(2,3)), l2_coef = l2_coef)
    bias_2 = Bias(np.random.normal(size=(2,1)), l2_coef = l2_coef)
    nn = Sequential([lin_1, bias_1, LeakyReLu(0.1), lin_2, bias_2, LeakyReLu(0.1)])
    ce = CrossEntropyLoss()

    print('Train accuracy:', accuracy(nn, train_data, train_labels))
    print('Test accuracy:', accuracy(nn, test_data, test_labels))

    batch_sampler = BatchSampler(train_data, train_labels, batch_size=5)

    # num_samples = train_data.shape[1]
    num_epochs = 1000
    learning_rate = 0.001

    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_wo_regularization = 0
        for (batch_samples, batch_labels) in batch_sampler:
            cur_batch_size = len(batch_labels)
            nn.zero_grad()
            for (sample, label) in zip(batch_samples.T, batch_labels):
                # NN forwarding of entire batches not yet implemented, so we just loop batches
                sample = np.expand_dims(sample, 1)
                activations = nn.forward(sample)
                loss = ce.forward(activations, label)
                probs = softmax(activations)
                total_loss += loss
                total_loss_wo_regularization += loss
                grad_by_input = nn.backward(ce.backward())
            nn.backward_regularization(rate=cur_batch_size)
            total_loss += cur_batch_size * nn.regularization_loss()
            nn.step_grad(learning_rate)
        print('-' * 20 + '\n' + f'Epoch: {epoch}' )
        print(f"Total loss: {total_loss / train_data.shape[1]}; " +
              f"Total loss w/o regularization: {total_loss_wo_regularization / train_data.shape[1]}; " + 
              f"Regularization loss: {len(train_labels) * nn.regularization_loss()}")
        print('Train accuracy:', accuracy(nn, train_data, train_labels))
        print('Test accuracy:', accuracy(nn, test_data, test_labels))

if __name__ == '__main__':
    main()
