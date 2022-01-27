from typing import Tuple, List, Any, Optional, Sequence, Dict, Callable
import numpy as np
import math
from layers import Model
import unittest

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
    if (labels is not None) and (len(labels) != data.shape[-1]):
        raise ValueError("Data and labels are of inconsistent sizes")
    num_samples = data.shape[-1]
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
        return [(data[..., index_set], labels[index_set])  for index_set in index_sets]
    else:
        return [data[..., index_set]  for index_set in index_sets]

def accuracy(model, data: np.ndarray, labels: Sequence) -> float:
    num_samples = data.shape[-1]
    activations = model.forward(data)
    correct = np.sum(predict_label(activations) == labels)
    return correct / num_samples

def softmax(vec: np.ndarray) -> np.ndarray:
    shifted_vec = vec - np.max(vec)
    shifted_exp = np.exp(shifted_vec)
    exp_sum = np.sum(shifted_exp)
    return shifted_exp / exp_sum


def predict_label(activations: np.ndarray) -> np.ndarray:
    return np.argmax(activations, axis=0)

class BatchSampler:
    def __init__(self, data: np.ndarray, labels: Optional[Sequence] = None, batch_size: int = 1):
        if (labels is not None) and (len(labels) != data.shape[-1]):
            raise ValueError("Data and labels are of inconsistent sizes")
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return self.data.shape[-1]

    def __iter__(self):
        indices = np.argsort(np.random.uniform(size=len(self)))
        cur = 0
        while cur < len(self):
            next_cur = cur + self.batch_size
            indices_chunk = indices[cur:next_cur]
            if self.labels is not None:
                yield (self.data[..., indices_chunk], self.labels[indices_chunk])
            else:
                yield self.data[..., indices_chunk]
            cur = next_cur

def image_to_str(sample, threshold=10, shape=(28,28), marker_filled='x', marker_empty=' '):
    row_strings = []
    sample = sample.reshape(*shape)
    for row in range(sample.shape[0]):
        row_strings.append(''.join( (marker_filled if pix >= threshold else marker_empty)  for pix in sample[row,:]))
    return '\n'.join(row_strings)



class TestCaseExtended(unittest.TestCase):
    # I don't want to import pytest, so it's easier to reimplement basic assertion helpers
    def assertAllClose(self, first, second, tolerance = 1e-5):
        self.assertTrue(np.all(np.abs(first - second) <= tolerance))

class TestUtils(TestCaseExtended):
    def test_predict_label(self):
        self.assertEqual(predict_label(np.array([1,3,2,4,-5,0])), 3)

    def test_softmax(self):
        self.assertAllClose(softmax(np.array([5,5,5,5])), softmax(np.array([0.25, 0.25, 0.25, 0.25])))
        self.assertAllClose(softmax(np.array([5,7,2,-5])), softmax(np.array([105, 107, 102, 95])))
        self.assertAllClose(softmax(np.array([5,7,2,-5])), softmax(np.array([-95, -93, -98, -105])))
        self.assertAllClose(np.sum(softmax(np.array([5,15,-3,12]))), 1)

    def test_train_test_indices(self):
        first, second = train_test_indices(num_samples=10, sample_set_sizes=[3,7])
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 7)
        self.assertEqual(len(set(first) & set(second)), 0)
        self.assertEqual(sorted(set(first) | set(second)), list(range(10)))

    def test_train_test_indices_several_groups(self):
        first, second, third = train_test_indices(num_samples=10, sample_set_sizes=[3,5,2])
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 5)
        self.assertEqual(len(third), 2)
        self.assertEqual(len(set(first) & set(second)), 0)
        self.assertEqual(len(set(first) & set(third)), 0)
        self.assertEqual(len(set(second) & set(third)), 0)
        self.assertEqual(sorted(set(first) | set(second) | set(third)), list(range(10)))

    def test_train_test_indices_too_many(self):
        with self.assertRaises(Exception):
            first, second = train_test_indices(num_samples=10, sample_set_sizes=[3,8])


if __name__ == '__main__':
    unittest.main()

