import copy
import logging
import numpy as np
from layers import Model, CrossEntropyLoss
from typing import Dict

logger = logging.getLogger('minitorch')

def numeric_gradients(model: Model, sample: np.ndarray, label: int, step: float = 0.00001) -> Dict[str, np.ndarray]:
    loss = CrossEntropyLoss()
    parameters = model.parameters()
    gradients = {}
    cls = type(model)
    model_1 = cls.from_parameters(copy.deepcopy(parameters))
    model_2 = cls.from_parameters(copy.deepcopy(parameters))
    for param_name, weights in parameters.items():
        logger.info(f"Numeric gradient calculation for {param_name}...")
        weights_1 = weights.copy()
        weights_2 = weights.copy()
        idx_iter = np.nditer(weights, flags=['multi_index'])
        gradient = np.zeros_like(weights)
        for _ in idx_iter:
            idx = idx_iter.multi_index
            weights_1[idx] = weights[idx] - step
            weights_2[idx] = weights[idx] + step
            model_1.update_parameters({param_name: weights_1})
            model_2.update_parameters({param_name: weights_2})
            loss_1 = loss.forward(model_1.forward(sample), label)
            loss_2 = loss.forward(model_2.forward(sample), label)
            gradient[idx] = (loss_2 - loss_1) / (2 * step)
            # restore values
            weights_1[idx] = weights[idx]
            weights_2[idx] = weights[idx]
            model_1.update_parameters({param_name: weights_1})
            model_2.update_parameters({param_name: weights_2})
        gradients[param_name] = gradient
    return gradients

def backprop_gradients(model: Model, sample: np.ndarray, label: int):
    loss = CrossEntropyLoss()
    loss.forward(model.forward(sample), label)
    model.backward(loss.backward())
    return {param_name: layer.grad  for param_name, layer in model.variable_layers().items()}

def check_gradients(model: Model, sample: np.ndarray, label: int, tolerance: float = 1e-4) -> bool:
    loss = CrossEntropyLoss()
    num_gradients = numeric_gradients(model, sample, label)
    bp_gradients = backprop_gradients(model, sample, label)

    for param_name in model.variable_layers():
        discrepancy = np.max(np.abs(bp_gradients[param_name] - num_gradients[param_name]))
        if discrepancy > tolerance:
            logger.warning(f"Gradient checking failed for {param_name}, discrepancy: {discrepancy}.")
            return False
    return True
