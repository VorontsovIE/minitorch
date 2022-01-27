import copy
import logging
import numpy as np
from layers import Model, CrossEntropyLoss, SumOverBatch
from typing import Dict

logger = logging.getLogger('minitorch')

def numeric_gradients(model: Model, batch: np.ndarray, labels: np.ndarray, step: float = 0.00001) -> Dict[str, np.ndarray]:
    loss = CrossEntropyLoss()
    summer = SumOverBatch()
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
            loss_1 = summer.forward(loss.forward(model_1.forward(batch), labels))
            loss_2 = summer.forward(loss.forward(model_2.forward(batch), labels))
            gradient[idx] = (loss_2 - loss_1) / (2 * step)
            # restore values
            weights_1[idx] = weights[idx]
            weights_2[idx] = weights[idx]
            model_1.update_parameters({param_name: weights_1})
            model_2.update_parameters({param_name: weights_2})
        gradients[param_name] = gradient
    return gradients

def backprop_gradients(model: Model, batch: np.ndarray, labels: np.ndarray):
    loss = CrossEntropyLoss()
    summer = SumOverBatch()
    summer.forward(loss.forward(model.forward(batch), labels))
    model.backward(loss.backward(summer.backward(1.0)))
    return {param_name: layer.grad  for param_name, layer in model.variable_layers().items()}

def check_gradients(model: Model, batch: np.ndarray, labels: np.ndarray, tolerance: float = 1e-4) -> bool:
    # Note: here we can use loss as a parameter, or even use lambda model, batch, labels: loss(model, batch, labels)
    #   but it complicates code. Moreover, when loss computation is done via AST, an interface will be changed inevitably.
    loss = CrossEntropyLoss()
    num_gradients = numeric_gradients(model, batch, labels)
    bp_gradients = backprop_gradients(model, batch, labels)

    for param_name in model.variable_layers():
        discrepancy = np.max(np.abs(bp_gradients[param_name] - num_gradients[param_name]))
        if discrepancy > tolerance:
            logger.warning(f"Gradient checking failed for {param_name}, discrepancy: {discrepancy}.")
            return False
    return True
