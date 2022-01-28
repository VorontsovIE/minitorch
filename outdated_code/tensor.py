from __future__ import annotations
import numpy as np
from functools import cached_property
from typing import Sequence, Union

# from utils import unwrap_lines

def unwrap_lines(text):
    return ' '.join(map(str.strip, text.splitlines()))

class CalculationNode:
    @property
    def shape(self) -> Sequence[int]:
        "Output shape"
        raise NotImplementedError()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def zero_grad(self) -> None:
        raise NotImplementedError()

    def forward(self) -> np.ndarray:
        '''
        Recursively calculate node value.
        Node can store inputs or derived values necessary to calculate gradient.
        It's convenient to make a cached property for calculated(forwarded) input arguments.
        '''
        raise NotImplementedError()

    def backward(self, grad: np.ndarray) -> None:
        '''
        Push gradient backward.
        `grad` is a gradient wrt output activations of shape (output_activations, batch).
        It's usually necessary to first calculate value forward to know the point 
        at which a gradient is calculated. But it's done automatically.
        '''
        raise NotImplementedError()

    def __add__(self, other: CalculationNode) -> Add:
        if isinstance(other, CalculationNode):
            if self.shape != other.shape:
                # ToDo: add shape broadcasting
                raise ValueError(f"Operator __add__ cannot sum nodes of shapes {self.shape} and {other.shape}")
            return Add(self, other)
        else:
            raise ValueError(f"Operator __add__({type(self)}, {type(other)}) is not defined")

    def __sub__(self, other: CalculationNode) -> Subtract:
        if isinstance(other, CalculationNode):
            if self.shape != other.shape:
                raise ValueError(f"Operator __sub__ cannot subtract nodes of shapes {self.shape} and {other.shape}")
            return Subtract(self, other)
        else:
            raise ValueError(f"Operator __sub__({type(self)}, {type(other)}) is not defined")

    def __mul__(self, other: Union[int, float]) -> NumberMultiply:
        # ToDo: add matrix multiplication
        if isinstance(other, int) or isinstance(other, float):
            return NumberMultiply(matrix=self, number=other)
        else:
            raise ValueError(f"Operator __mul__({type(self)}, {type(other)}) is not defined")

    def __rmul__(self, other: Union[int, float]) -> NumberMultiply:
        if isinstance(other, int) or isinstance(other, float):
            return NumberMultiply(matrix=self, number=other)
        else:
            raise ValueError(f"Operator __rmul__({type(self)}, {type(other)}) is not defined")

    def __matmul__(self, other: CalculationNode) -> MatrixMultiply:
        if isinstance(other, CalculationNode):
            if (self.ndim == 2) and (other.ndim == 2) and (self.shape[1] == other.shape[0]):
                return MatrixMultiply(self, other)
            else:
                raise ValueError(f"Operator __matmul__ cannot multiply nodes of shapes {self.shape} and {other.shape}")
        else:
            raise ValueError(f"Operator __matmul__({type(self)}, {type(other)}) is not defined")

    def __neg__(self) -> Negate:
        return Negate(self)


class Tensor(CalculationNode):
    def __init__(self, data, differentiable=True):
        self.data = np.array(data, dtype=float)
        self.differentiable = differentiable
        self.zero_grad()
    
    def __repr__(self):
        matrix_repr = unwrap_lines(repr(self.data))
        matrix_repr = matrix_repr[6:-1] # "array([...])" -> "[...]"
        return  f'Tensor({matrix_repr})'

    __str__ = __repr__

    def zero_grad(self) -> None:
        if self.differentiable:
            self.grad = np.zeros_like(self.data, dtype=float)
    
    def step_grad(self, rate) -> None:
        if self.differentiable:
            self.data -= rate * self.grad

    @property
    def shape(self):
        return self.data.shape

    def forward(self) -> np.ndarray:
        return self.data

    def backward(self, grad):
        if self.differentiable:
            self.grad += grad


class UnaryOperation(CalculationNode):
    def __init__(self, arg: CalculationNode):
        self.arg = arg

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.arg)})'

    def zero_grad(self) -> None:
        self.arg.zero_grad()

    def step_grad(self, rate) -> None:
        self.arg.step_grad(rate)


class BinaryOperation(CalculationNode):
    def __init__(self, first: CalculationNode, second: CalculationNode):
        self.first = first
        self.second = second

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.first)}, {repr(self.second)})'

    def zero_grad(self) -> None:
        self.first.zero_grad()
        self.second.zero_grad()

    def step_grad(self, rate) -> None:
        self.first.step_grad(rate)
        self.second.step_grad(rate)


class Add(BinaryOperation):
    def __init__(self, first: CalculationNode, second: CalculationNode):
        if first.shape != second.shape:
            raise ValueError(f"Can't add nodes of shapes {first.shape} and {second.shape}")
        super().__init__(first, second)

    @property
    def shape(self) -> Sequence[int]:
        return self.first.shape

    def forward(self) -> np.ndarray:
        return self.first.forward() + self.second.forward()

    def backward(self, grad: np.ndarray) -> None:
        self.first.backward(grad)
        self.second.backward(grad)


class Subtract(BinaryOperation):
    def __init__(self, first: CalculationNode, second: CalculationNode):
        if first.shape != second.shape:
            raise ValueError(f"Can't subtract nodes of shapes {first.shape} and {second.shape}")
        super().__init__(first, second)

    @property
    def shape(self) -> Sequence[int]:
        return self.first.shape

    def forward(self) -> np.ndarray:
        return self.first.forward() - self.second.forward()

    def backward(self, grad: np.ndarray) -> None:
        self.first.backward(grad)
        self.second.backward(-grad)


class MatrixMultiply(BinaryOperation):
    def __init__(self, first: CalculationNode, second: CalculationNode):
        if (first.ndim != 2) or (second.ndim != 2):
            raise ValueError(f"Can't multiply non-2D nodes of shapes {first.shape} and {second.shape}")
        if first.shape[1] != second.shape[0]:
            raise ValueError(f"Can't multiply nodes of shapes {first.shape} and {second.shape}")
        super().__init__(first, second)

    @property
    def shape(self) -> Sequence[int]:
        return (self.first.shape[0], self.second.shape[1])

    def forward(self) -> np.ndarray:
        return self.first.forward() @ self.second.forward()

    @cached_property
    def first_value(self) -> np.ndarray:
        return self.first.forward()

    @cached_property
    def second_value(self) -> np.ndarray:
        return self.second.forward()

    def backward(self, grad: np.ndarray) -> None:
        self.first.backward(grad @ self.second_value.T)
        self.second.backward(self.first_value.T @ grad)


class NumberMultiply(CalculationNode):
    def __init__(self, matrix: CalculationNode, number: Union[int, float]):
        self.matrix = matrix
        self.number = number

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.matrix)}, {repr(self.number)})'

    def zero_grad(self) -> None:
        self.matrix.zero_grad()

    def step_grad(self, rate) -> None:
        self.matrix.step_grad(rate)

    @property
    def shape(self) -> Sequence[int]:
        return self.matrix.shape

    def forward(self) -> np.ndarray:
        return self.number * self.matrix.forward()

    def backward(self, grad: np.ndarray) -> None:
        self.matrix.backward(self.number * grad)


class Negate(UnaryOperation):
    @property
    def shape(self) -> Sequence[int]:
        return self.arg.shape

    def forward(self) -> np.ndarray:
        return -self.arg.forward()

    def backward(self, grad: np.ndarray) -> None:
        self.arg.backward(-grad)

class SumOverBatch(UnaryOperation):
    @property
    def shape(self) -> Sequence[int]:
        return self.arg.shape[:-1]

    @cached_property
    def arg_value(self) -> np.ndarray:
        return self.arg.forward()

    def forward(self) -> np.ndarray:
        return self.arg_value.sum(axis=-1)

    def backward(self, grad: np.ndarray) -> None:
        batch_size = self.arg_value.shape[-1]
        self.arg.backward(np.ones(shape=(1, batch_size)) * grad)


class CrossEntropyLoss(CalculationNode):
    '''
    Cross entropy of soft-maxed input against a certain class label.
    NOTE! Input vector is raw values, not probabilites vector.
          Softmax of input is calculated inside of CE-loss.
    '''
    def __init__(self, activations: CalculationNode, labels: np.ndarray):
        if labels.ndim != 1:
            raise ValueError(f"Labels should be 1-dim but was of shape {labels.shape}.")
        if activations.ndim != 2:
            raise ValueError(f"Activations should be 2-dim but was of shape {activations.shape}.")
        if activations.shape[-1] != len(labels):
            raise ValueError(f"Labels of shape {labels.shape} and activations of shape {activations.shape} are inconsistent.")
        self.activations = activations
        self.labels = labels

    def zero_grad(self) -> None:
        self.activations.zero_grad()

    def step_grad(self, rate) -> None:
        self.activations.step_grad(rate)

    @cached_property
    def activations_val(self) -> np.ndarray:
        return self.activations.forward()

    @cached_property
    def shifted_vec(self) -> np.ndarray:
        vec = self.activations_val
        return vec - np.max(vec, axis=0)

    @cached_property
    def shifted_exp(self) -> np.ndarray:
        return np.exp(self.shifted_vec)

    @cached_property
    def exp_sum(self) -> np.ndarray:
        return np.sum(self.shifted_exp, axis=0)

    @property
    def shape(self) -> Sequence[int]:
        return (1, self.activations.shape[-1])

    def forward(self) -> np.ndarray:
        # H = -p*log(q) where p is actual class distribution, q is predicted distribution
        #   = -1*log(q_l) = -log(SoftMax_l(v_1..v_n))  where l is label
        #   = -log(\exp{v_l} / (\sum_k \exp{v_k}) )
        #   = log(\sum_k \exp{v_k}) - v_l
        #   = log(\exp{m}\sum_k \exp{v_k - m}) - v_l   for any m
        #   = log(\sum_k \exp{v_k - m}) + m - v_l      we put m = max(v_k)
        #   = log(\sum_k \exp{w_k}) - w_l              where w_i = v_i - m
        labels = self.labels
        batch_size = len(labels)
        result = np.log(self.exp_sum) - self.shifted_vec[self.labels, np.arange(batch_size)]
        return result.reshape(1, batch_size)

    def backward(self, grad: np.ndarray) -> None:
        # dH/dx_i = \exp{v_i} / (\sum_k\exp{v_k}) - \delta_{i,l} =
        #         = \exp{v_i - m} / (\sum_k\exp{v_k - m}) - \delta_{i,l}
        #         = \exp{w_i} / (\sum_k\exp{w_k}) - \delta_{i,l}
        batch_size = len(self.labels)
        if grad.shape != (1, batch_size):
            raise ValueError(f'Gradient should be of shape (1, batch_size={batch_size}) but was {grad.shape}')
        result = self.shifted_exp / self.exp_sum
        result[self.labels, np.arange(batch_size)] -= 1
        input_grad = result * grad  # shape (num_classes, batch_size)
        self.activations.backward(input_grad)


class L2_norm(UnaryOperation):
    @property
    def shape(self) -> Sequence[int]:
        return (1,)

    @cached_property
    def arg_value(self) -> np.ndarray:
        return self.arg.forward()

    def forward(self) -> np.ndarray:
        return np.sum(self.arg_value ** 2)

    def backward(self, grad: np.ndarray) -> None:
        self.arg.backward(2 * self.arg_value)


class ReLu(UnaryOperation):
    @property
    def shape(self) -> Sequence[int]:
        return self.arg.shape

    @cached_property
    def vec_value(self) -> np.ndarray:
        return self.arg.forward()

    def forward(self) -> np.ndarray:
        vec = self.vec_value
        if vec.ndim != 2:
            raise ValueError(f'vec should be 2-dim (activations, batch_size) but was {vec.shape}')
        return (vec > 0) * vec

    def backward(self, grad: np.ndarray) -> None:
        if grad.ndim != 2:
            raise ValueError(f'grad should be 2-dim (activations, batch_size) but was {grad.shape}')
        return (self.vec_value > 0) * grad


class LeakyReLu(UnaryOperation):
    def __init__(self, alpha):
        self.alpha = alpha

    @property
    def shape(self) -> Sequence[int]:
        return self.arg.shape

    @cached_property
    def vec_value(self) -> np.ndarray:
        return self.arg.forward()

    def forward(self) -> np.ndarray:
        vec = self.vec_value
        if vec.ndim != 2:
            raise ValueError(f'vec should be 2-dim (activations, batch_size) but was {vec.shape}')
        return np.where(vec > 0, 1, self.alpha) * vec

    def backward(self, grad: np.ndarray) -> None:
        vec = self.vec_value
        if grad.ndim != 2:
            raise ValueError(f'grad should be 2-dim (activations, batch_size) but was {grad.shape}')
        return np.where(vec > 0, 1, self.alpha) * grad

# Посмотреть, что там с размерностью Bias
# Сложение с числом ?

