import os
import struct
import numpy as np
import matplotlib.pyplot as plt



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
        self.X_train, self.y_train = load_mnist('mnist/', kind='train')
        print('Rows: %d, columns: %d' % (self.X_train.shape[0], self.X_train.shape[1]))
        self.X_test, self.y_test = load_mnist('mnist/', kind='t10k')
        print('Rows: %d, columns: %d' % (self.X_test.shape[0], self.X_test.shape[1]))

    def save_CSV(self):
        np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
        np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
        X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')
        y_train = np.genfromtxt('train_labels.csv', dtype=int, delimiter=',')

        np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
        np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')
        X_test = np.genfromtxt('test_img.csv', dtype=int, delimiter=',')
        y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')

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
