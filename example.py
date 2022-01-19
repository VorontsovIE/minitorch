import numpy as np
import random
from utils import BatchSampler, train_test_split, softmax, predict_label, accuracy
from layers import Sequential, Linear, Bias, ReLu, LeakyReLu, CrossEntropyLoss
from mnist_loader import read_labels, read_images
import logging

FORMAT = '%(levelname)s: %(asctime)s — %(message)s'
logging.basicConfig(format=FORMAT, level='DEBUG')
logger = logging.getLogger('minitorch')

def main():
    seed = 137
    random.seed(seed)
    np.random.seed(seed)

    train_labels_fn = 'mnist_data/train-labels-idx1-ubyte'
    train_images_fn = 'mnist_data/train-images-idx3-ubyte'
    test_labels_fn  = 'mnist_data/t10k-labels-idx1-ubyte'
    test_images_fn  = 'mnist_data/t10k-images-idx3-ubyte'

    input_size = 28*28 # size of image
    num_classes = 10
    train_val_labels = read_labels(train_labels_fn)
    test_labels  = read_labels(test_labels_fn)
    # realign images pixels into a single vector
    train_val_data = read_images(train_images_fn).reshape(-1, input_size)
    test_data  = read_images(test_images_fn).reshape(-1, input_size)

    (train_data, train_labels), (validation_data, validation_labels) = train_test_split(
        data=train_val_data,
        labels=train_val_labels,
        sample_set_rates=(0.9, 0.1)
    )
    # One should use metrics on validation for hyperparameter tuning (I didn't)
    logger.info('Data loaded')

    hidden_size_1 = 300
    l2_coef = 0.01 / (input_size + hidden_size_1 + num_classes)
    lin_1  = Linear(np.random.normal(size=(hidden_size_1, input_size)), l2_coef = l2_coef)
    bias_1 = Bias(np.random.normal(size=(hidden_size_1, 1)), l2_coef = l2_coef)
    lin_2  = Linear(np.random.normal(size=(num_classes, hidden_size_1)), l2_coef = l2_coef)
    bias_2 = Bias(np.random.normal(size=(num_classes, 1)), l2_coef = l2_coef)
    nn = Sequential([lin_1, bias_1, LeakyReLu(0.1), lin_2, bias_2, LeakyReLu(0.1)])
    ce = CrossEntropyLoss()
    batch_size = 16
    learning_rate = 0.00001
    # num_epochs = 25
    num_epochs = 2

    logger.info(f'(before training) Train accuracy: {accuracy(nn, train_data, train_labels)}')
    logger.info(f'(before training) Validation accuracy: {accuracy(nn, validation_data, validation_labels)}')
    logger.info(f'(before training) Test accuracy: {accuracy(nn, test_data, test_labels)}')

    batch_sampler = BatchSampler(train_data, train_labels, batch_size=batch_size)

    # num_samples = train_data.shape[1]

    for epoch in range(num_epochs):
        logger.info(f'Epoch started: {epoch}')
        total_loss = 0
        num_samples = 0
        total_loss_wo_regularization = 0
        for (batch_samples, batch_labels) in batch_sampler:
            cur_batch_size = 0
            nn.zero_grad()
            for (sample, label) in zip(batch_samples, batch_labels):
                # NN forwarding of entire batches not yet implemented, so we just loop batches
                sample = np.expand_dims(sample, 1)
                activations = nn.forward(sample)
                loss = ce.forward(activations, label)
                probs = softmax(activations)
                if np.isnan(loss) or np.isinf(loss) or np.any(np.isnan(probs) | np.isinf(probs)):
                    continue
                cur_batch_size += 1
                total_loss += loss
                total_loss_wo_regularization += loss
                grad_by_input = nn.backward(ce.backward())
            nn.backward_regularization(rate=cur_batch_size)
            total_loss += cur_batch_size * nn.regularization_loss()
            num_samples += cur_batch_size
            nn.step_grad(learning_rate)
        logger.info(f'Epoch finished: {epoch}')
        logger.info(
          f"Total loss: {total_loss / num_samples}; " +
          f"Total loss w/o regularization: {total_loss_wo_regularization / num_samples}; " +
          f"Regularization loss: {num_samples * nn.regularization_loss()}"
        )
        logger.info(f'Train accuracy: {accuracy(nn, train_data, train_labels)}')
        logger.info(f'Validation accuracy: {accuracy(nn, validation_data, validation_labels)}')

    logger.info(f'Test accuracy: {accuracy(nn, test_data, test_labels)}')

if __name__ == '__main__':
    main()
