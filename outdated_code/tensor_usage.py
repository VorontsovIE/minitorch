import numpy as np
import tensor

# pos — label 1
# neg — label 0
train_pos = np.random.normal(size=(10,2), loc=(5.0, 1.0), scale=1.0).transpose()
train_neg = np.random.normal(size=(10,2), loc=(-1.0, 10.0), scale=1.0).transpose()
train_data = np.hstack([train_pos, train_neg])
train_labels = np.hstack([1 * np.ones(train_pos.shape[1], dtype=int), 0 * np.ones(train_neg.shape[1], dtype=int)])

test_pos = np.random.normal(size=(10,2), loc=(5.0, 1.0), scale=1.0).transpose()
test_neg = np.random.normal(size=(10,2), loc=(-1.0, 10.0), scale=1.0).transpose()
test_data = np.hstack([test_pos, test_neg])
test_labels = np.hstack([1 * np.ones(test_pos.shape[1], dtype=int), 0 * np.ones(test_neg.shape[1], dtype=int)])


# l2_coef = 0.01
# lin_1 = Linear(np.random.normal(size=(3,2)), l2_coef = l2_coef)
# bias_1 = Bias(np.random.normal(size=(3,1)), l2_coef = l2_coef)
# lin_2 = Linear(np.random.normal(size=(2,3)), l2_coef = l2_coef)
# bias_2 = Bias(np.random.normal(size=(2,1)), l2_coef = l2_coef)
# nn = Sequential([lin_1, bias_1, LeakyReLu(0.1), lin_2, bias_2, LeakyReLu(0.1)])
# ce = CrossEntropyLoss()

def basic_loss(data, labels):
    return  tensor.SumOverBatch(
                tensor.CrossEntropyLoss(
                    lin_2 @ lin_1 @ tensor.Tensor(train_data, differentiable=False),
                    train_labels
                )
            )

def regularization_loss():
    return tensor.L2_norm(lin_1) + tensor.L2_norm(lin_2)

lin_1 = tensor.Tensor(np.random.normal(size=(3,2)))
lin_2 = tensor.Tensor(np.random.normal(size=(2,3)))
for _ in range(200):
    loss = basic_loss(train_data, train_labels) + 0.1 * regularization_loss()
    print(loss.forward())
    loss.backward(1.0)
    loss.step_grad(0.000001)

loss_test = basic_loss(test_data, test_labels) + 0.1 * regularization_loss()
print(loss_test.forward())
