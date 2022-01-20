import numpy as np
from utils import unwrap_lines

class CalculationNode:
    # def __init__(differentiable=True):
        # self.differentiable = differentiable
        # self.differential = np.zeros_like(self.data)

    @property
    def shape(self):
        raise NotImplementedError()

    def value(self):
        raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, CalculationNode):
            if self.shape != other.shape:
                # ToDo: add shape broadcasting
                raise ValueError(f"Operator __add__ cannot sum nodes of shapes {self.shape} and {other.shape}")
            return Add(self, other)
        else:
            raise ValueError(f"Operator __add__({type(self)}, {type(other)}) is not defined")

    def __sub__(self, other):
        if isinstance(other, CalculationNode):
            if self.shape != other.shape:
                raise ValueError(f"Operator __sub__ cannot subtract nodes of shapes {self.shape} and {other.shape}")
            return Subtract(self, other)
        else:
            raise ValueError(f"Operator __sub__({type(self)}, {type(other)}) is not defined")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return NumberMultiply(matrix=self, number=other)
        else:
            raise ValueError(f"Operator __mul__({type(self)}, {type(other)}) is not defined")

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return NumberMultiply(matrix=self, number=other)
        else:
            raise ValueError(f"Operator __rmul__({type(self)}, {type(other)}) is not defined")

    def __matmul__(self, other):
        if isinstance(other, CalculationNode):
            if (len(self.shape) == 2) and (len(other.shape) == 2) and (self.shape[1] == other.shape[0]):
                return MatrixMultiply(self, other)
            else:
                raise ValueError(f"Operator __matmul__ cannot multiply nodes of shapes {self.shape} and {other.shape}")
        else:
            raise ValueError(f"Operator __matmul__({type(self)}, {type(other)}) is not defined")

    def __neg__(self):
        return Negate(self)


class Tensor(CalculationNode):
    def __init__(self, data):
        super().__init__(differentiable=differentiable)
        self.data = np.array(data)
        self.differential = np.zeros_like(self.data)
    
    def __repr__(self):
        matrix_repr = unwrap_lines(repr(self.data))
        matrix_repr = matrix_repr[6:-1] # "array([...])" -> "[...]"
        return  f'Tensor({matrix_repr})'

    __str__ = __repr__
    
    def zero_grad(self):
        self.differential = np.zeros_like(self.data)

    def step_grad(self, rate=0.001):
        self.data -= rate * self.differential

    @property
    def shape(self):
        return self.data.shape

    def value(self):
        return self.data

    def backward(self, mul_l, mul_r):
        self.differential += mul_l


class UnaryOperation(CalculationNode):
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.arg)})'


class BinaryOperation(CalculationNode):
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.first)}, {repr(self.second)})'


class Add(BinaryOperation):
    def __init__(self, first, second):
        if first.shape != second.shape:
            raise ValueError(f"Can't add nodes of shapes {first.shape} and {second.shape}")
        super().__init__(first, second)

    @property
    def shape(self):
        return self.first.shape

    def value(self):
        return self.first.value() + self.second.value()

    def backward(self, mul_l, mul_r):
        self.first.backward(mul_l, mul_r)
        self.second.backward(mul_l, mul_r)


class Subtract(BinaryOperation):
    def __init__(self, first, second):
        if first.shape != second.shape:
            raise ValueError(f"Can't subtract nodes of shapes {first.shape} and {second.shape}")
        super().__init__(first, second)

    @property
    def shape(self):
        return self.first.shape

    def value(self):
        return self.first.value() - self.second.value()

    def backward(self, mul_l, mul_r):
        self.first.backward(mul_l, mul_r)
        self.second.backward(-mul_l, mul_r)


class MatrixMultiply(BinaryOperation):
    def __init__(self, first, second):
        if (len(first.shape) != 2) or (len(second.shape) != 2):
            raise ValueError(f"Can't multiply non-2D nodes of shapes {first.shape} and {second.shape}")
        if first.shape[1] != second.shape[0]:
            raise ValueError(f"Can't multiply nodes of shapes {first.shape} and {second.shape}")
        super().__init__(first, second)

    @property
    def shape(self):
        return self.first.shape

    def value(self):
        return self.first.value() @ self.second.value()

    def backward(self, mul_l, mul_r):
        self.first().backward(mul_l, mul_r * self.second.value()) # dimension??????
        self.second().backward(mul_l * self.first.value(), mul_r)


class NumberMultiply(CalculationNode):
    def __init__(self, matrix, number):
        self.matrix = matrix
        self.number = number

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.matrix)}, {repr(self.number)})'

    @property
    def shape(self):
        return self.matrix.shape

    def value(self):
        return self.matrix.value() * self.number

    def backward(self, mul_l, mul_r):
        self.matrix.backward(mul * self.number)


class Negate(UnaryOperation):
    @property
    def shape(self):
        return self.arg.shape

    def value(self):
        return -self.arg.value()

    def backward(self, mul_l, mul_r):
        self.arg.backward(-mul)
