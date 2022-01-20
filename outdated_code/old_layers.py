class Layer:
    @property
    def in_dim(self):
        raise NotImplementedError()

    @property
    def out_dim(self):
        raise NotImplementedError()

    def forward(self, data):
        raise NotImplementedError()

class Linear(Layer):
    def __init__(self, weights=None, in_dim=None, out_dim=None):
        if weights:
            weights = np.array(weights)
            if len(weights.shape) != 2:
                raise ValueError("Linear layer weight matrix should be 2-dimensional")
            if (in_dim  is not None) and (in_dim  != weights.shape[1]):
                raise ValueError("`in_dim` is inconsistent with `weights`")
            if (out_dim is not None) and (out_dim != weights.shape[0]):
                raise ValueError("`out_dim` is inconsistent with `weights`")
            self.weights = weights
        else:
            if in_dim and out_dim:
                self.weights = np.random.randn(out_dim, in_dim)
            else:
                raise ValueError("Either `weights` or (`in_dim` + `out_dim`) should be specified")

    @property
    def in_dim(self):
        self.weights.shape[1]

    @property
    def out_dim(self):
        self.weights.shape[0]

    def forward(self, data):
        if (self.in_dim, ) != data.shape[:-1]:
            raise ValueError(f"`data` should be of shape `({self.in_dim}, batch_size)` but was of shape {data.shape}")
        return np.matmul(data, self.weights)

