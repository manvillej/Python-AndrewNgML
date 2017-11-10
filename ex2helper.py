import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def plotData(X, Y):
    positiveExamples = plt.scatter(
                    np.extract(Y == 1, X[:, 1]),
                    np.extract(Y == 1, X[:, 2]),
                    label="y=1",
                    marker='o',
                    color='b',
                    s=10)

    negativeExamples = plt.scatter(
                    np.extract(Y == 0, X[:, 1]),
                    np.extract(Y == 0, X[:, 2]),
                    label="y=0",
                    marker='x',
                    color='r',
                    s=10)

    plt.legend(handles=[positiveExamples, negativeExamples], loc='lower left')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, x, y):
    m = x.shape[0]
    z = sigmoid(np.matmul(x, theta))
    J = -1 / m * (np.sum(np.log(z) * y + np.log(1 - z) * (1 - y)))
    return J


def gradient(theta, x, y):
    m = x.shape[0]
    z = sigmoid(np.matmul(x, theta))
    grad = np.matmul(x.transpose(), z - y) / m
    return grad


def optimize(theta, x, y):
    return op.minimize(
                    fun=costFunction,
                    x0=theta,
                    args=(x, y),
                    method='TNC',
                    jac=gradient)


def predict(theta, x):
    p = sigmoid(np.matmul(x, theta))

    predictions = np.zeros(p.shape)
    predictions[np.where(p >= .5)] = 1

    return predictions


def plotDecisionBoundary(theta, X, Y):
    plotData(X, Y)
    if (X.shape[1] <= 3):
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 2]), max(X[:, 2])])
        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])
        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros([u.shape[0], v.shape[0]])
        for i in range(0, z.shape[0]):
            for j in range(0, z.shape[1]):
                r = np.array([u[i], v[j]])
                r = r[np.newaxis, :]
                z[i][j] = np.matmul(mapFeatures(r), theta)
        z = z.transpose()
        plt.contour(u, v, z, 0, colors='k')


def mapFeatures(X):
    degrees = 6
    [m, n] = X.shape
    mapped_X = np.ones([X.shape[0], 1])
    for i in range(1, degrees + 1):
        for j in range(0, i + 1):
            r = np.multiply(np.power(X[:, 0], i - j), np.power(X[:, 1], j))
            mapped_X = np.append(mapped_X, r[:, np.newaxis], axis=1)

    return mapped_X


def costFunctionReg(theta, x, y, lambdaVal):
    if (y.ndim > 1):
        y = np.squeeze(y)
    m = x.shape[0]
    if (y.shape[0] != m):
        raise ValueError('Y & X are not compatible: X.shape = {} &  y.shape = {}'.format(
            X.shape,
            y.shape))

    z = sigmoid(np.matmul(x, theta))

    pos = np.multiply(np.log(z), y)
    neg = np.multiply(np.log(1 - z), (1 - y))

    J = -1 / m * (np.sum(np.add(pos, neg)))

    reg = np.ones(theta.shape)
    reg[0] = 0
    reg = (lambdaVal / (2 * m)) * np.sum(np.multiply(reg, np.power(theta, 2)))

    return J + reg


def gradientReg(theta, x, y, lambdaVal):
    if (y.ndim > 1):
        y = np.squeeze(y)
    m = x.shape[0]
    if (y.shape[0] != m):
        raise ValueError('Y & X are not compatible: X.shape = {} &  y.shape = {}'.format(
            X.shape,
            y.shape))

    z = sigmoid(np.matmul(x, theta))

    grad = np.matmul(x.transpose(), np.subtract(z, y)) / m

    reg = np.ones(theta.shape)
    reg[0] = 0
    reg = (lambdaVal / (m)) * np.multiply(reg, theta)

    return np.add(grad, reg)


def optimizeReg(theta, x, y, lambdaVal):
    return op.minimize(
                    fun=costFunctionReg,
                    x0=theta,
                    args=(x, y, lambdaVal),
                    method='TNC',
                    jac=gradientReg)
