import os
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt

from nn_mlp import NeuralNetMLP
from mlp_gc import MLPGradientCheck

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


class mnist(object):
    def __init__(self):
        print('Loading Training data...')
        self.X_train, self.y_train = load_mnist('mnist/', kind='train')
        print('Loading Testing data...')
        self.X_test, self.y_test = load_mnist('mnist/', kind='t10k')

        print('Rows: %d, columns: %d' % (self.X_train.shape[0], self.X_train.shape[1]))
        print('Rows: %d, columns: %d' % (self.X_test.shape[0], self.X_test.shape[1]))

    def plot_digits(self):
        X_train = self.X_train
        y_train = self.y_train

        fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
        ax = ax.flatten()
        for i in range(10):
            img = X_train[y_train == i][0].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        # plt.savefig('./figures/mnist_all.png', dpi=300)
        plt.show()

    def plot_7s(self):
        X_train = self.X_train
        y_train = self.y_train

        fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
        ax = ax.flatten()
        for i in range(25):
            img = X_train[y_train == 7][i].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        # plt.savefig('./figures/mnist_7.png', dpi=300)
        plt.show()

    def train_NN_MLP(self, epochs=500):
        X_train = self.X_train
        y_train = self.y_train

        print('Initializing Neural Net MLP ...')
        self.nn = NeuralNetMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=epochs,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50,
                  shuffle=True,
                  random_state=1)
        print('Training Neural Net MLP ...')
        self.nn.fit(X_train, y_train, print_progress=True)
        print('Neural Net MLP Fit SUCCESSFUL')

    def plot_all_NN_MLP(self):
        nn = self.nn

        plt.figure(1)
        plt.plot(range(len(nn.cost_)), nn.cost_)
        plt.ylim([0, 2000])
        plt.ylabel('Cost')
        plt.xlabel('Epochs * 50')
        plt.tight_layout()
        if not os.path.exists('./figures/cost.png'):
            plt.savefig('./figures/cost.png', dpi=300)
        batches = np.array_split(range(len(nn.cost_)), 1000)
        cost_ary = np.array(nn.cost_)
        cost_avgs = [np.mean(cost_ary[i]) for i in batches]
        plt.figure(2)
        plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
        plt.ylim([0, 2000])
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        plt.tight_layout()
        if not os.path.exists('./figures/cost2.png'):
            plt.savefig('./figures/cost2.png', dpi=300)
        plt.show()

    def test_NN_MLP(self):
        y_train = self.y_train
        X_train = self.X_train
        y_test = self.y_test
        X_test = self.X_test

        y_train_pred = self.nn.predict(X_train)
        y_test_pred = self.nn.predict(X_test)

        if sys.version_info < (3, 0):
            acc_train = ((np.sum(y_train == y_train_pred, axis=0)).astype('float') /
                   X_train.shape[0])
            acc_test = ((np.sum(y_test == y_test_pred, axis=0)).astype('float') /
                                      X_test.shape[0])
        else:
            acc_train = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
            acc_test = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]

        print('Training accuracy: %.2f%%' % (acc_train * 100))
        print('Test accuracy: %.2f%%' % (acc_test * 100))

    def mlp_gc_fit(self):
        X_train = self.X_train
        y_train = self.y_train
        nn_check = MLPGradientCheck(n_output=10,
                            n_features=X_train.shape[1],
                            n_hidden=10,
                            l2=0.0,
                            l1=0.0,
                            epochs=10,
                            eta=0.001,
                            alpha=0.0,
                            decrease_const=0.0,
                            minibatches=1,
                            shuffle=False,
                            random_state=1)
        nn_check.fit(X_train[:5], y_train[:5], print_progress=True)
