import functools
import math
import numpy as np

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


class CrossEntropyLoss:
    def forward(self, vec, label):
        self.input = vec
        self.label = label
        return -math.log(vec[label])
    def backward(self):
        result = np.zeros_like(self.input)
        result[self.label] = -1.0 / self.input[self.label]
        return result



def predict_label(probs):
    return np.argmax(probs.squeeze())

pos = np.random.normal(size=(10,2), loc=(5.0, 1.0), scale=1.0).transpose()
neg = np.random.normal(size=(10,2), loc=(1.0, 5.0), scale=1.0).transpose()
data = np.hstack([pos, neg])
labels = np.hstack([1 * np.ones(pos.shape[1], dtype=int), 0 * np.ones(neg.shape[1], dtype=int)])
# pos — label 1
# neg — label 0

lin_1 = Linear(np.random.normal(size=(2,2)), l2_coef = 1)
sm = SoftMax()
nn = Sequential([lin_1, sm])
ce = CrossEntropyLoss()

num_samples = data.shape[1]

lin_1.weights = np.array([[ 0.2310601,  -1.35196172],[ 0.49964013,  1.6349675 ]])
print(lin_1.weights)
for _ in range(5):
    nn.zero_grad()
    total_loss = 0
    for idx in range(num_samples):
        sample = np.expand_dims(data[:, idx], 1)
        probs = nn.forward(sample)
        loss = ce.forward(probs, labels[idx])
        if np.isnan(loss):
            print(f"(rejected) sample: {sample.squeeze()}; probs: {probs.squeeze()}; label: {label}; loss: {loss};")
            continue
        print(f"sample: {sample.squeeze()}; probs: {probs.squeeze()}; label: {label}; loss: {loss};")
        total_loss += loss
        grad_by_input = nn.backward(ce.backward())
        print(f"grad: {lin_1.grad}")
    nn.backward_regularization(rate=num_samples)
    nn.step_grad(0.00001)
    print(f"(after regularization) grad: {lin_1.grad}")
    total_loss += num_samples * nn.regularization_loss()
    print(f"Total loss: {total_loss / data.shape[1]}")
    print([predict_label(nn.forward( np.expand_dims(data[:,idx], 1) ))  for idx in range(num_samples)])

[predict_label(nn.forward( np.expand_dims(data[:,idx], 1) ))  for idx in range(num_samples)]

# idx = 1; sample = np.expand_dims(data[:, idx], 1); probs = nn.forward(sample); loss = ce.forward(probs, labels[idx]);print(f"------\nidx: {idx}");print(f"probs: {probs.squeeze()}"); print(f"loss: {loss}"); print(f"backward: {ce.backward().squeeze()}");

