"""
==============================================================
Scikit-learn-compatible Kernel Discriminant Analysis.
基于 Kernel LDA + KNN 的人脸识别
使用 Kernel Discriminant Analysis 做特征降维
使用 K-Nearest-Neighbor 做分类

数据:
    人脸图像来自于 Olivetti faces data-set from AT&T (classification)
    数据集包含 40 个人的人脸图像, 每个人都有 10 张图像
    我们只使用其中标签(label/target)为 0 和 1 的前 2 个人的图像

算法:
    需要自己实现基于 RBF Kernel 的 Kernel Discriminant Analysis 用于处理两个类别的数据的特征降维
    代码的框架已经给出, 需要学生自己补充 KernelDiscriminantAnalysis 的 fit() 和 transform() 函数的内容
==============================================================
"""
# License: BSD 3 clause

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import linalg
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.discriminant_analysis import _cov
from sklearn.utils.multiclass import unique_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _class_means(X, y):
    """Compute class means.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    means : array-like, shape (n_classes, n_features)
        Class means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means


def _class_cov(X, y):
    """Compute class covariance matrix.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    cov : array-like, shape (n_features, n_features)
        Class covariance matrix.
    """
    classes = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        cov += np.atleast_2d(_cov(Xg))
    return cov


class KernelDiscriminantAnalysis(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Kernel Discriminant Analysis.

    Parameters
    ----------
    n_components: integer.
                  The dimension after transform.
    gamma: float.
           Parameter to RBF Kernel
    lmb: float (>= 0.0), default=0.001.
         Regularization parameter

    """

    def __init__(self, n_components, gamma, lmb=0.001):
        self.n_components = n_components
        self.gamma = gamma
        self.lmb = lmb
        self.X = None # 用于存放输入的训练数据的 X
        self.K = None # 用于存放训练数据 X 产生的 Kernel Matrix
        self.M = None # 用于存放 Kernel LDA 最优化公式中的 M
        self.N = None # 用于存放 Kernel LDA 最优化公式中的 N
        self.EigenVectors = None # 用于存放 Kernel LDA 最优化公式中的 M 对应的广义特征向量, 每一列为一个特征向量, 按照对应特征值大小排序

    def _solve_eigen(self, X, y):
        """Eigenvalue solver.

        The eigenvalue solver computes the optimal solution of the Rayleigh
        coefficient (basically the ratio of between class scatter to within
        class scatter). This solver supports both classification and
        dimensionality reduction (with optional shrinkage).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y)

        Sw = self.covariance_  # within scatter
        St = _cov(X)  # total scatter
        Sb = St - Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals)
                                                 )[::-1][:self._max_components]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means_, self.coef_.T)))

    def _solve_svd(self, X, y):
        """SVD solver.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.means_ = _class_means(X, y)

        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group, :]
            Xc.append(Xg - self.means_[idx])

        self.priors = None
        self.tol = 1e-4
        if self.priors is None:  # estimate priors from sample
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            self.priors_ = np.bincount(y_t) / float(len(y))
        self.xbar_ = np.dot(self.priors_, self.means_)

        Xc = np.concatenate(Xc, axis=0)

        # 1) within (univariate) scaling by with classes std-dev
        std = Xc.std(axis=0)
        # avoid division by zero in normalization
        std[std == 0] = 1.
        fac = 1. / (n_samples - n_classes)

        # 2) Within variance scaling
        X = np.sqrt(fac) * (Xc / std)
        # SVD of centered (within)scaled data
        U, S, V = linalg.svd(X, full_matrices=False)

        rank = np.sum(S > self.tol)
        # Scaling of within covariance is: V' 1/S
        scalings = (V[:rank] / std).T / S[:rank]

        # 3) Between variance scaling
        # Scale weighted centers
        X = np.dot(((np.sqrt((n_samples * self.priors_) * fac)) *
                    (self.means_ - self.xbar_).T).T, scalings)
        # Centers are living in a space with n_classes-1 dim (maximum)
        # Use SVD to find projection in the space spanned by the
        # (n_classes) centers
        _, S, V = linalg.svd(X, full_matrices=0)

        self.explained_variance_ratio_ = (S**2 / np.sum(
            S**2))[:self._max_components]
        rank = np.sum(S > self.tol * S[0])
        self.scalings_ = np.dot(scalings, V.T[:, :rank])
        coef = np.dot(self.means_ - self.xbar_, self.scalings_)
        self.intercept_ = (-0.5 * np.sum(coef ** 2, axis=1) +
                           np.log(self.priors_))
        self.coef_ = np.dot(coef, self.scalings_.T)
        self.intercept_ -= np.dot(self.xbar_, self.coef_.T)

    def fit_LDA(self, X, y):
        """Fit KDA model.

        Parameters
        ----------
        X: numpy array of shape [n_samples, n_features]
           Training set.
        y: numpy array of shape [n_samples]
           Target values. Only works for 2 classes with label/target 0 and 1.

        Returns
        -------
        self

        """
        X, y = check_X_y(X, y, ensure_min_samples=2, estimator=self,
                         dtype=[np.float64, np.float32])
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)

        if n_samples == n_classes:
            raise ValueError("The number of samples must be more "
                             "than the number of classes.")

        # Maximum number of components no matter what n_components is
        # specified:
        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                self._max_components = max_components
            else:
                self._max_components = self.n_components

        self._solve_svd(X, y)

        if self.classes_.size == 2:  # treat binary case as a special case
            self.coef_ = np.array(self.coef_[1, :] - self.coef_[0, :], ndmin=2,
                                  dtype=X.dtype)
            self.intercept_ = np.array(self.intercept_[1] - self.intercept_[0],
                                       ndmin=1, dtype=X.dtype)
        return self

    def transform_LDA(self, X):
        """Transform data with the trained KernelLDA model.

        Parameters
        ----------
        X_test: numpy array of shape [n_samples, n_features]
           The input data.

        Returns
        -------
        y_pred: array-like, shape (n_samples, n_components)
                Transformations for X.

        """
        check_is_fitted(self, ['xbar_', 'scalings_'], all_or_any=any)

        X = check_array(X)
        X_new = np.dot(X - self.xbar_, self.scalings_)

        return X_new[:, :self._max_components]

    def RBF_kernel(self, X1, X2):
        n1,d1 = X1.shape
        n2,d2 = X2.shape
        if d1 != d2:
            print('RBF dimensions do not match.')
            exit(1)

        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                x1 = X1[i]
                x2 = X2[j]
                # d = np.linalg.norm(x1 - x2)**2
                d = np.sum(np.square(x1-x2))
                K[i,j] = np.exp(-self.gamma*d)

        return K.T

    def fit(self, X, y):
        """Fit KDA model.

        Parameters
        ----------
        X: numpy array of shape [n_samples, n_features]
           Training set.
        y: numpy array of shape [n_samples]
           Target values. Only works for 2 classes with label/target 0 and 1.

        Returns
        -------
        self

        """
        self.X = X
        X1 = X[y == 0, :]
        X2 = X[y == 1, :]
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        n = n1 + n2

        K1 = self.RBF_kernel(X, X1)
        K2 = self.RBF_kernel(X, X2)

        M1 = np.mean(K1, axis=0)
        M2 = np.mean(K2, axis=0)
        M12 = M1 - M2
        M = np.matmul(np.array([M12]).T, np.array([M12]))

        P1 = np.eye(n1)-1/n1*np.ones((n1,n1))
        P2 = np.eye(n2)-1/n2*np.ones((n2,n2))
        N1 = np.matmul(np.matmul(K1.T,P1),K1)
        N2 = np.matmul(np.matmul(K2.T,P2),K2)
        N = N1 + N2

        self.lmb = 1e-3
        N_reg = N +self.lmb*np.eye(n)  # np.mean(N) = 0.005, np.np.mean(np.abs(N)) = 0.03,self.lmb = 0.001
        N_inv = np.linalg.inv(N_reg)
        # N_inv = np.linalg.inv(N)  # unhealthy value
        # N_inv = np.linalg.pinv(N) # works
        A = N_inv*M
        e_vals, e_vecs = np.linalg.eig(A)
        indices = np.argsort(e_vals)
        indices = indices[::-1]
        e_vecs = e_vecs[:,indices]
        self.w = e_vecs[:,:self.n_components]

        return self

    def transform(self, X_test):
        """Transform data with the trained KernelLDA model.

        Parameters
        ----------
        X_test: numpy array of shape [n_samples, n_features]
           The input data.

        Returns
        -------
        y_pred: array-like, shape (n_samples, n_components)
                Transformations for X.

        """
        Kx = self.RBF_kernel(self.X, X_test)
        # X_new = np.matmul(Kx, self.w)
        X_new = np.dot(Kx, self.w)

        return X_new


def test_LDA(X_train, y_train, X, y):
    # lda = LinearDiscriminantAnalysis(n_components=1, solver='eigen')
    lda = LinearDiscriminantAnalysis(n_components=2, solver='svd')
    lda.fit(X_train, y_train)
    X_new = lda.transform(X)
    X_embed = np.zeros((X_new.shape[0],2))
    X_embed[:,0] = X_new[:,0]
    X_embed[:,1] = 1
    plt.scatter(X_embed[:, 0], X_embed[:, 1], c=y, marker='o')
    plt.show()

    lda2 = KernelDiscriminantAnalysis(n_components=2, gamma=5e-6)
    lda2.fit_LDA(X_train, y_train)
    X_new2 = lda2.transform_LDA(X)
    X_embed2 = np.zeros((X_new2.shape[0],2))
    X_embed2[:,0] = X_new2[:,0]
    X_embed2[:,1] = 0
    # plt.figure()
    plt.scatter(X_embed2[:, 0], X_embed2[:, 1], c=y, marker='o')
    plt.show()


def test_KLDA(X, y, X_train, y_train, X_test, y_test, n_neighbors):
    # Reduce dimension to 2 with KernelDiscriminantAnalysis
    # can adjust the value of 'gamma' as needed.
    kda = make_pipeline(StandardScaler(),
                        KernelDiscriminantAnalysis(n_components=2, gamma = 5e-6))

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the method's model
    kda.fit(X_train, y_train)

    # Fit a nearest neighbor classifier on the embedded training set
    X_train_kda = kda.transform(X_train)
    knn.fit(X_train_kda, y_train)

    # Compute the nearest neighbor accuracy on the embedded test set
    X_test_kda = kda.transform(X_test)
    acc_knn = knn.score(X_test_kda, y_test)

    # Embed the data set in 2 dimensions using the fitted model
    X_embed = kda.transform(X)

    # Plot the projected points and show the evaluation score
    plt.figure()
    plt.scatter(X_embed[:, 0], X_embed[:, 1], c=y, s=30, cmap='Set1')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format('kda', n_neighbors, acc_knn))
    plt.show()


def main():
    # 指定 KNN 中最近邻的个数 (k 的值)
    n_neighbors = 3

    # 设置随机数种子让实验可以复现
    random_state = 0

    # 现在人脸数据集
    faces = fetch_olivetti_faces()
    targets = faces.target

    # show sample images
    images = faces.images[targets < 2] # save images

    features = faces.data  # features
    targets = faces.target # targets

    fig = plt.figure() # create a new figure window
    for i in range(20): # display 20 images
        # subplot : 4 rows and 5 columns
        img_grid = fig.add_subplot(4, 5, i+1)
        # plot features as image
        img_grid.imshow(images[i], cmap='gray')

    plt.show()

    # Prepare data, 只限于处理类别 0 和 1 的人脸
    X, y = faces.data[targets < 2], faces.target[targets < 2]

    # Split into train/test
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.5, stratify=y,
                         random_state=random_state)

    # test_LDA(X_train, y_train, X, y)

    test_KLDA(X, y, X_train, y_train, X_test, y_test, n_neighbors)


if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    t_all = t_end - t_start
    print('whole time: {:.2f} min'.format(t_all / 60.))