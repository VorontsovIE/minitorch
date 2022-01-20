from typing import Tuple, List, Any, Optional, Sequence, Dict, Callable
import numpy as np
import math
from layers import Model

def train_test_indices(num_samples: int, sample_set_sizes: List[int]) -> List[ np.ndarray]:
    if not all((x >= 0 and isinstance(x, int)) for x in sample_set_sizes):
        raise ValueError("sample set sizes should be non-negative")
    if sum(sample_set_sizes) > num_samples:
        raise ValueError(f"Total set size {num_samples} should be less or equal to a sum of part sizes {sample_set_sizes}.")
    indices = np.argsort(np.random.uniform(size=num_samples))
    result = []
    cur = 0
    for size in sample_set_sizes:
        next_cur = cur + size
        result.append(indices[cur:next_cur])
        cur = next_cur
    return result

def train_test_split(data: np.ndarray,
                    labels: Optional[np.ndarray]=None,
                    sample_set_sizes: Optional[List[int]] = None,
                    sample_set_rates: Optional[List[float]] = None):
    if (labels is not None) and (len(labels) != data.shape[0]):
        raise ValueError("Data and labels are of inconsistent sizes")
    num_samples = data.shape[0]
    if sample_set_rates:
        if sample_set_sizes:
            raise ValueError("Specify either rates or numbers of sample sets, not both")
        else:
            sample_set_sizes = [math.floor(num_samples * rate)  for rate in sample_set_rates]
    else:
        if not sample_set_sizes:
            raise ValueError("Specify either rates or numbers of sample sets")
    index_sets = train_test_indices(num_samples, sample_set_sizes)
    if labels is not None:
        return [(data[index_set], labels[index_set])  for index_set in index_sets]
    else:
        return [data[index_set]  for index_set in index_sets]

def accuracy(model, data: np.ndarray, labels: Sequence) -> float:
    num_samples = data.shape[0]
    correct = 0
    for idx in range(num_samples):
        sample = np.expand_dims(data[idx], 1)
        activations = model.forward(sample)
        if predict_label(activations) == labels[idx]:
            correct += 1
    return correct / num_samples

def softmax(vec: np.ndarray) -> np.ndarray:
    shifted_vec = vec - np.max(vec)
    shifted_exp = np.exp(shifted_vec)
    exp_sum = np.sum(shifted_exp)
    return shifted_exp / exp_sum


def predict_label(activations: np.ndarray) -> int:
    return np.argmax(activations.squeeze()).astype(int)

class BatchSampler:
    def __init__(self, data: np.ndarray, labels: Optional[Sequence] = None, batch_size: int = 1):
        if (labels is not None) and (len(labels) != data.shape[0]):
            raise ValueError("Data and labels are of inconsistent sizes")
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        indices = np.argsort(np.random.uniform(size=len(self)))
        cur = 0
        while cur < len(self):
            next_cur = cur + self.batch_size
            indices_chunk = indices[cur:next_cur]
            if self.labels is not None:
                yield (self.data[indices_chunk], self.labels[indices_chunk])
            else:
                yield self.data[indices_chunk]
            cur = next_cur

def gradients(model: Model, loss: Callable[[Model], float], step: float = 0.00001) -> Dict[str, np.ndarray]:
    cls = type(model)
    parameters = model.parameters()
    gradients = {}
    for key, weights in parameters.items():
        idx_iter = np.nditer(weights, flags=['multi_index'])
        gradient = np.zeros_like(weights)
        for _ in idx_iter:
            idx = idx_iter.multi_index

            parameters_1 = parameters.copy()
            weights_1 = weights.copy()
            weights_1[idx] = weights[idx] - step
            parameters_1[key] = weights_1
            model_1 = cls.from_parameters(parameters_1)

            parameters_2 = parameters.copy()
            weights_2 = weights.copy()
            weights_2[idx] = weights[idx] - step
            parameters_2[key] = weights_2
            model_2 = cls.from_parameters(parameters_2)

            gradient[idx] = (loss(model_2) - loss(model_1)) / (2 * step)
        gradients[key] = gradient
    return gradients

def print_image(sample, threshold=10, shape=(28,28), marker_filled='x', marker_empty=' '):
    sample = sample.reshape(*shape)
    for row in range(sample.shape[0]):
        print(''.join( (marker_filled if pix >= threshold else marker_empty)  for pix in sample[row,:]))
