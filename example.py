import numpy as np
import random
from utils import BatchSampler, train_test_split, softmax, predict_label, accuracy
from layers import Sequential, Linear, Bias, ReLu, LeakyReLu, CrossEntropyLoss

def main():
    seed = 138
    random.seed(seed)
    np.random.seed(seed)

    # train_labels_fn = 'mnist_data/train-labels-idx1-ubyte'
    # train_images_fn = 'mnist_data/train-images-idx3-ubyte'
    # test_labels_fn  = 'mnist_data/t10k-labels-idx1-ubyte'
    # test_images_fn  = 'mnist_data/t10k-images-idx3-ubyte'

    # train_labels = read_labels(train_labels_fn)
    # train_images = read_images(train_images_fn)
    # test_labels = read_labels(test_labels_fn)
    # test_images = read_images(test_images_fn)


    # pos — label 1
    # neg — label 0
    train_pos = np.random.normal(size=(10, 2), loc=(5.0, 1.0), scale=1.0)
    train_neg = np.random.normal(size=(10, 2), loc=(1.0, 5.0), scale=1.0)
    train_data = np.vstack([train_pos, train_neg])
    train_labels = np.vstack([1 * np.ones((train_pos.shape[0], 1), dtype=int), 0 * np.ones((train_neg.shape[0], 1), dtype=int)])

    test_pos = np.random.normal(size=(10, 2), loc=(5.0, 1.0), scale=1.0)
    test_neg = np.random.normal(size=(10, 2), loc=(1.0, 5.0), scale=1.0)
    test_data = np.vstack([test_pos, test_neg])
    test_labels = np.vstack([1 * np.ones((test_pos.shape[0],1), dtype=int), 0 * np.ones((test_neg.shape[0],1), dtype=int)])

    l2_coef = 0.01
    lin_1 = Linear(np.random.normal(size=(3,2)), l2_coef = l2_coef)
    bias_1 = Bias(np.random.normal(size=(3,1)), l2_coef = l2_coef)
    lin_2 = Linear(np.random.normal(size=(2,3)), l2_coef = l2_coef)
    bias_2 = Bias(np.random.normal(size=(2,1)), l2_coef = l2_coef)
    nn = Sequential([lin_1, bias_1, LeakyReLu(0.1), lin_2, bias_2, LeakyReLu(0.1)])
    ce = CrossEntropyLoss()

    print('Train accuracy:', accuracy(nn, train_data, train_labels))
    print('Test accuracy:', accuracy(nn, test_data, test_labels))

    batch_sampler = BatchSampler(train_data, train_labels, batch_size=5)

    # num_samples = train_data.shape[1]
    num_epochs = 1000
    learning_rate = 0.001

    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_wo_regularization = 0
        for (batch_samples, batch_labels) in batch_sampler:
            cur_batch_size = len(batch_labels)
            nn.zero_grad()
            for (sample, label) in zip(batch_samples, batch_labels):
                # NN forwarding of entire batches not yet implemented, so we just loop batches
                sample = np.expand_dims(sample, 1)
                activations = nn.forward(sample)
                loss = ce.forward(activations, label)
                probs = softmax(activations)
                total_loss += loss
                total_loss_wo_regularization += loss
                grad_by_input = nn.backward(ce.backward())
            nn.backward_regularization(rate=cur_batch_size)
            total_loss += cur_batch_size * nn.regularization_loss()
            nn.step_grad(learning_rate)
        print('-' * 20 + '\n' + f'Epoch: {epoch}' )
        print(f"Total loss: {total_loss / train_data.shape[1]}; " +
              f"Total loss w/o regularization: {total_loss_wo_regularization / train_data.shape[1]}; " + 
              f"Regularization loss: {len(train_labels) * nn.regularization_loss()}")
        print('Train accuracy:', accuracy(nn, train_data, train_labels))
        print('Test accuracy:', accuracy(nn, test_data, test_labels))

if __name__ == '__main__':
    main()
