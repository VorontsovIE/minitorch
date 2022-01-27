import random
import os
import argparse
import logging
import numpy as np
from utils import BatchSampler, train_test_split, softmax, predict_label, accuracy, image_to_str
from gradient_checking import check_gradients
from layers import CrossEntropyLoss, SumOverBatch
from mnist_loader import read_labels, read_images
from mnist_clf import ImageClassificationModel

FORMAT = '%(levelname)s: %(asctime)s — %(message)s'
logging.basicConfig(format=FORMAT, level='DEBUG')
logger = logging.getLogger('minitorch')

class Main:
    input_size = 28 * 28 # size of image
    skip_train = False

    def run(self):
        self.configure()

        if self.args.seed:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)

        if self.args.operating_mode == 'train-test':
            self.load_or_init_model()
            self.load_data()
            self.train_val_split()
            if not self.skip_train:
                self.train()
            test_accuracy = accuracy(self.model, self.test_data, self.test_labels)
            logger.info(f'Test accuracy: {test_accuracy}')
        else:
            self.load_or_init_model()
            self.load_data()
            logger.info('Gradient checking started...')
            idx = np.random.randint(self.train_val_data.shape[0])
            sample = np.expand_dims(self.train_val_data[idx], 1)
            label = self.train_val_labels[idx]
            print(f'Label: {label}')
            print(image_to_str(sample))
            if not check_gradients(self.model, sample, label, tolerance = 1e-4):
                logger.error(f"Gradient checking failed on train sample {idx}.")
            else:
                logger.info(f"Gradient checking passed on train sample {idx}.")

    def configure(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-epochs', metavar='EPOCHS', dest='num_epochs',
                            type=int, default=2, help='Number of epochs to train model.')
        parser.add_argument('--model-filename', metavar='FILENAME', dest='model_fn',
                            default='model.npz', help='File to load/store the model.')
        parser.add_argument('--seed', type=int, help='Random seed')
        parser.add_argument('--batch-size', dest='batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=0.00001, help='Learning rate')
        parser.add_argument('--operating-mode', dest='operating_mode',
                            choices=['train-test', 'gradient-checking'],
                            default='train-test', help='Specify operation to perform (train-test / gradient-checking).')
        self.args = parser.parse_args()

    def load_data(self):
        train_labels_fn = 'mnist_data/train-labels-idx1-ubyte'
        train_images_fn = 'mnist_data/train-images-idx3-ubyte'
        test_labels_fn  = 'mnist_data/t10k-labels-idx1-ubyte'
        test_images_fn  = 'mnist_data/t10k-images-idx3-ubyte'

        self.train_val_labels = read_labels(train_labels_fn)
        self.test_labels  = read_labels(test_labels_fn)
        # realign images pixels into a single vector
        self.train_val_data = read_images(train_images_fn).reshape(-1, self.input_size)
        self.test_data  = read_images(test_images_fn).reshape(-1, self.input_size)
        logger.info('Data loaded')

    def train_val_split(self):
        # I didn't tune model, but...
        # One should use metrics on validation for hyperparameter tuning, not on test dataset
        (self.train_data, self.train_labels), (self.val_data, self.val_labels) = train_test_split(
            data=self.train_val_data,
            labels=self.train_val_labels,
            sample_set_rates=(0.9, 0.1)
        )
    
    def load_or_init_model(self):
        model_fn = self.args.model_fn
        if os.path.exists(model_fn):
            self.model = ImageClassificationModel.load(model_fn)
            logger.info(f"Model loaded from {model_fn}.")
            logger.warning("Model won't be trained! We can't guarantee that train/val " +
                           "split at pretraining and training stages was the same. " +
                           "Thus the model will be overfitted and we won't even notice it.")
            self.skip_train = True
        else:
            self.model = ImageClassificationModel.initialize_random(input_size=self.input_size, hidden_size=300)
            logger.info(f"Initialized model with random data.")

    def train_single_epoch(self):
        model = self.model
        ce = CrossEntropyLoss()
        summer = SumOverBatch()
        total_loss = 0
        num_samples = 0
        total_loss_wo_regularization = 0
        for (batch_samples, batch_labels) in self.batch_sampler:
            model.zero_grad()
            activations = model.forward(batch_samples)
            losses = ce.forward(activations, batch_labels)
            loss = summer.forward(losses)
            total_loss += loss
            total_loss_wo_regularization += loss
            grad_by_input = model.backward(ce.backward(summer.backward(1.0)))
            cur_batch_size = len(batch_labels)
            model.backward_regularization(rate=cur_batch_size)
            total_loss += cur_batch_size * model.regularization_loss()
            num_samples += cur_batch_size
            model.step_grad(self.args.learning_rate)
        logger.info(f"Total loss: {total_loss / num_samples}")
        logger.debug(f"Total loss w/o regularization: {total_loss_wo_regularization / num_samples}")
        logger.debug(f"Regularization loss: {num_samples * model.regularization_loss()}")

    def train(self):
        model = self.model
        self.batch_sampler = BatchSampler(self.train_data, self.train_labels, batch_size=self.args.batch_size)

        train_accuracy = accuracy(model, self.train_data, self.train_labels)
        val_accuracy = accuracy(model, self.val_data, self.val_labels)
        test_accuracy = accuracy(model, self.test_data, self.test_labels)
        logger.info(f'(before training) Train accuracy: {train_accuracy}')
        logger.info(f'(before training) Validation accuracy: {val_accuracy}')
        logger.info(f'(before training) Test accuracy: {test_accuracy}')

        best_val_accuracy = 0

        for epoch in range(self.args.num_epochs):
            logger.info(f'Epoch started: {epoch}')
            self.train_single_epoch()
            logger.info(f'Epoch finished: {epoch}')
            train_accuracy = accuracy(model, self.train_data, self.train_labels)
            val_accuracy = accuracy(model, self.val_data, self.val_labels)
            logger.info(f'Train accuracy: {train_accuracy}')
            logger.info(f'Validation accuracy: {val_accuracy}')
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model.store(self.args.model_fn)
                logger.info(f"Store model to {self.args.model_fn}.")

if __name__ == '__main__':
    Main().run()
