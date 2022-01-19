from typing import Tuple, List, Any, Optional, Sequence
import numpy as np
# import numpy.typing as npt
import math

def train_test_indices(num_samples: int, sample_set_sizes: List[int]) -> List[ np.ndarray]:
    if not all((x >= 0 and isinstance(x, int)) for x in sample_set_sizes):
        raise ValueError("sample set sizes should be non-negative")
    if sum(sample_set_sizes) > num_samples:
        raise ValueError(f"Total set size {num_samples} should be less or equal to a sum of part sizes {sample_set_sizes}.")
    indices = np.argsort(np.random.uniform(size=num_samples))
    result = []
    cur = 0
    for size in sample_set_sizes:
        result.append(indices[cur : (cur + size)])
        cur += size
    return result

def train_test_split(data: np.ndarray,
                    labels: Optional[np.ndarray]=None,
                    sample_set_sizes: Optional[List[int]] = None,
                    sample_set_rates: Optional[List[float]] = None):
                    # sample_set_sizes = None,
                    # sample_set_rates = None):
    if (labels is not None) and (len(labels) != data.shape[1]):
        raise ValueError("Data and labels are of inconsistent sizes")
    num_samples = data.shape[1]
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
        return [(data[:, index_set], labels[index_set])  for index_set in index_sets]
    else:
        return [data[:, index_set]  for index_set in index_sets]


def unwrap_lines(text):
    return ' '.join(map(str.strip, text.splitlines()))

class BatchSampler:
    def __init__(self, data: np.ndarray, labels: Optional[Sequence] = None, batch_size=1):
        if (labels is not None) and (len(labels) != data.shape[1]):
            raise ValueError("Data and labels are of inconsistent sizes")
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return self.data.shape[1]

    def __iter__(self):
        indices = np.argsort(np.random.uniform(size=len(self)))
        cur = 0
        while cur < len(self):
            next_cur = cur + self.batch_size
            indices_chunk = indices[cur:next_cur]
            if self.labels is not None:
                yield (self.data[:, indices_chunk], self.labels[indices_chunk])
            else:
                yield self.data[:, indices_chunk]
            cur = next_cur
