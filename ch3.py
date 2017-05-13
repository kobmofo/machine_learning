import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from sklearn                import datasets
from sklearn                import __version__  as sklearn_version
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import accuracy_score

from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.linear_model   import Perceptron
from sklearn.linear_model   import LogisticRegression
from sklearn.svm            import SVC

from distutils.version      import LooseVersion as Version
from matplotlib.colors      import ListedColormap

if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split


def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def cost_1(z):
    return - np.log(sigmoid(z))

def cost_0(z):
    return - np.log(1 - sigmoid(z))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


class ch3(object):
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data[:, [2, 3]]
        self.y = iris.target

        # print('Class labels:', np.unique(self.y))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)

        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)

    def plot_perceptron(self):
        X_train_std = self.X_train_std
        X_test_std = self.X_test_std
        y_train = self.y_train
        y_test = self.y_test

        ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
        ppn.fit(X_train_std, y_train)

        y_pred = ppn.predict(X_test_std)
        print('Perceptron')
        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        plot_decision_regions(X=X_combined_std, y=y_combined,
                                    classifier=ppn, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('Perceptron')
        plt.tight_layout()
        if not os.path.exists('./figures/iris_perceptron_scikit.png'):
            plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
        plt.show()

    def plot_logistic_reg(self):
        X_train_std = self.X_train_std
        X_test_std = self.X_test_std
        y_train = self.y_train
        y_test = self.y_test

        lr = LogisticRegression(C=1000.0, random_state=0)
        lr.fit(X_train_std, y_train)

        y_pred = lr.predict(X_test_std)
        print('Logistic Regression')
        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        plot_decision_regions(X_combined_std, y_combined,
                              classifier=lr, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('Logistic Regression')
        plt.tight_layout()
        if not os.path.exists('./figures/logistic_regression.png'):
            plt.savefig('./figures/logistic_regression.png', dpi=300)
        plt.show()

    def plot_svc(self):
        X_train_std = self.X_train_std
        X_test_std = self.X_test_std
        y_train = self.y_train
        y_test = self.y_test

        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(X_train_std, y_train)

        y_pred = svm.predict(X_test_std)
        print('SVC')
        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))


        plot_decision_regions(X_combined_std, y_combined,
                              classifier=svm, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('SVC')
        plt.tight_layout()
        if not os.path.exists('./figures/support_vector_machine_linear.png'):
            plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
        plt.show()

    def plot_svc_rbf(self):
        X_train_std = self.X_train_std
        X_test_std = self.X_test_std
        y_train = self.y_train
        y_test = self.y_test

        svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
        svm.fit(X_train_std, y_train)

        y_pred = svm.predict(X_test_std)
        print('SVC RBF')
        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        plot_decision_regions(X_combined_std, y_combined,
                              classifier=svm, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('SVC RBF')
        plt.tight_layout()
        if not os.path.exists('./figures/support_vector_machine_rbf_iris_1.png'):
            plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
        plt.show()

    def plot_decision_tree(self):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
        tree.fit(X_train, y_train)

        y_pred = tree.predict(X_test)
        print('Decision Tree')
        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

        X_combined = np.vstack((X_train, X_test))
        y_combined = np.hstack((y_train, y_test))
        plot_decision_regions(X_combined, y_combined,
                              classifier=tree, test_idx=range(105, 150))

        plt.xlabel('petal length [cm]')
        plt.ylabel('petal width [cm]')
        plt.legend(loc='upper left')
        plt.title('Decision Tree')
        plt.tight_layout()
        if not os.path.exists('./figures/decision_tree_decision.png'):
            plt.savefig('./figures/decision_tree_decision.png', dpi=300)
        plt.show()

    def plot_random_forest(self):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        forest = RandomForestClassifier(criterion='entropy',
                                        n_estimators=10,
                                        random_state=1,
                                        n_jobs=2)
        forest.fit(X_train, y_train)

        y_pred = forest.predict(X_test)
        print('Random Forest')
        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

        X_combined = np.vstack((X_train, X_test))
        y_combined = np.hstack((y_train, y_test))
        plot_decision_regions(X_combined, y_combined,
                              classifier=forest, test_idx=range(105, 150))

        plt.xlabel('petal length [cm]')
        plt.ylabel('petal width [cm]')
        plt.legend(loc='upper left')
        plt.title('Random Forest')
        plt.tight_layout()
        if not os.path.exists('./figures/random_forest.png'):
            plt.savefig('./figures/random_forest.png', dpi=300)
        plt.show()

    def plot_kneighbors(self):
        X_train_std = self.X_train_std
        X_test_std = self.X_test_std
        y_train = self.y_train
        y_test = self.y_test

        knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
        knn.fit(X_train_std, y_train)

        y_pred = knn.predict(X_test_std)
        print('K_neighbors')
        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        plot_decision_regions(X_combined_std, y_combined,
                              classifier=knn, test_idx=range(105, 150))

        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('K Nearest Neighbors')
        plt.tight_layout()
        if not os.path.exists('./figures/k_nearest_neighbors.png'):
            plt.savefig('./figures/k_nearest_neighbors.png', dpi=300)
        plt.show()
