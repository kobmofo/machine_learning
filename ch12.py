import os
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
        if not any ([os.path.exists('mnist/train_img.csv'), os.path.exists('mnist/train_labels.csv'), os.path.exists('mnist/test_img.csv'), os.path.exists('mnist/test_labels.csv')]):
            self._load_mnist()

        print('Loading X_train ...')
        self.X_train = np.genfromtxt('mnist/train_img.csv', dtype=int, delimiter=',')
        print('Loading y_train ...')
        self.y_train = np.genfromtxt('mnist/train_labels.csv', dtype=int, delimiter=',')
        print('Loading X_test ...')
        self.X_test = np.genfromtxt('mnist/test_img.csv', dtype=int, delimiter=',')
        print('Loading y_test ...')
        self.y_test = np.genfromtxt('mnist/test_labels.csv', dtype=int, delimiter=',')
        print('Rows: %d, columns: %d' % (self.X_train.shape[0], self.X_train.shape[1]))
        print('Rows: %d, columns: %d' % (self.X_test.shape[0], self.X_test.shape[1]))

    def _load_mnist(self):
        print('converting mnist to csv')
        X_train, y_train = load_mnist('mnist/', kind='train')
        np.savetxt('mnist/train_img.csv', X_train, fmt='%i', delimiter=',')
        np.savetxt('mnist/train_labels.csv', y_train, fmt='%i', delimiter=',')
        X_test, y_test = load_mnist('mnist/', kind='t10k')
        np.savetxt('mnist/test_img.csv', X_test, fmt='%i', delimiter=',')
        np.savetxt('mnist/test_labels.csv', y_test, fmt='%i', delimiter=',')


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

    def plot_nn_mlp(self):
        X_train = self.X_train
        y_train = self.y_train

        nn = NeuralNetMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=100,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50,
                  shuffle=True,
                  random_state=1)
        nn.fit(X_train, y_train, print_progress=True)
        # plt.plot(range(len(nn.cost_)), nn.cost_)
        # plt.ylim([0, 2000])
        # plt.ylabel('Cost')
        # plt.xlabel('Epochs * 50')
        # plt.tight_layout()
        # # plt.savefig('./figures/cost.png', dpi=300)
        # plt.show()
        batches = np.array_split(range(len(nn.cost_)), 1000)
        cost_ary = np.array(nn.cost_)
        cost_avgs = [np.mean(cost_ary[i]) for i in batches]
        plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
        plt.ylim([0, 2000])
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        plt.tight_layout()
        #plt.savefig('./figures/cost2.png', dpi=300)
        plt.show()

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
