import numpy as np
import ex2helper as helper2
import matplotlib.pyplot as plt
from sklearn import svm


def plotData(X, y, **kwargs):
    '''
    predict the value for the provided linear regression model
    '''
    addBias = kwargs.pop(
        'addBias',
        False)

    Xtemp = X

    if(addBias):
        Xtemp = np.insert(
            Xtemp,
            0,
            np.ones(
                Xtemp.shape[0]),
            axis=1)

    helper2.plotData(Xtemp, y)


def visualizeBoundary(X, y, svc, h=0.02):

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(
        np.arange(
            x_min,
            x_max,
            h),
        np.arange(
            y_min,
            y_max,
            h))

    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(
        xx,
        yy,
        Z,
        [0],
        colors='k',
        linewidths=1)

    plotData(X, y, addBias=True)

    # Support vectors indicated in plot by vertical lines
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')


def linearKernel(X1, X2):
    return np.dot(x1, x2)


def gaussianKernel(X1, X2, **kwargs):
    sigma = kwargs.pop('sigma', 1)

    if(isinstance(X1, np.ndarray)):
        X1 = X1.flatten()

    if(isinstance(X2, np.ndarray)):
        X2 = X2.flatten()

    euc = np.sum(
        np.power(X1, 2) +
        np.power(X2, 2) -
        (2*X1*X2))

    sim = np.exp(-euc/(2*sigma**2))

    return sim


def dataset3Params(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    score = 0

    possible = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

    for testC in possible:
        for testSigma in possible:
            gamma = np.power(testSigma, -2.)

            # train the model
            model = svm.SVC(
                C=testC,
                kernel='rbf',
                gamma=gamma)

            model.fit(X, y.flatten())

            # evalue model on cross validation set
            testScore = model.score(Xval, yval)

            # if better pair is found
            if(score < testScore):
                print('winning score: {}'.format(testScore))
                score = testScore
                sigma = testSigma
                C = testC

    return [C, sigma]
