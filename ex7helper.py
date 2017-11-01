import numpy as np
import matplotlib.pyplot as plt
import ex3helper as helper3


def findClosestCentroid(X, centroids):
    """FINDCLOSESTCENTROIDS computes the centroid memberships for every example
    idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

    # stack centroids so that X and centroid can be subtracted
    centroidMatrix = np.tile(
        centroids,
        (X.shape[0], 1, 1))

    # stack X so that X and centroid can be subtracted
    Xmatrix = np.tile(
        X,
        (centroids.shape[0], 1, 1))

    # swap axes so Xmatrix and centroidMatrix have matching axes
    Xmatrix = np.swapaxes(Xmatrix, 0, 1)

    '''
    How to check the structure of Xmatrix
    for i in range(3):
        print(Xmatrix[:3, i, :].flatten())
    print(X[:3, :].flatten())
    '''

    # get distance of each value
    results = np.power(Xmatrix-centroidMatrix, 2)
    # sum difference
    results = np.sum(results, axis=2)
    # get position of minimum
    results = np.argmin(results, axis=1)

    return results


def computeCentroids(X, idx, k):
    """COMPUTECENTROIDS returns the new centroids by computing the means of the data points assigned to each centroid.
    centroids = computeCentroids(X, idx, K) returns the new centroids by
    computing the means of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    """

    centroids = np.zeros((k, X.shape[1]))

    for i in range(k):
        centroids[i, :] = np.mean(X[idx == i, :], axis=0)

    return centroids


def featureNormalize(X):
    """normalizes X (X-mean)/sigma"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X-mean)/std


def PCA(Xnorm):
    """returns PCA"""
    m = Xnorm.shape[0]
    sigma = np.matmul(Xnorm.T, Xnorm,)/m
    return np.linalg.svd(sigma, full_matrices=True)


def drawLines(p1, p2, color):
    """DRAWLINE Draws a line from point p1 to point p2
    DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
    current figure
    """

    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)


def projectData(X, U, K):
    """PROJECTDATA Computes the reduced data representation when projecting only on to the top k eigenvectors
    Z = projectData(X, U, K) computes the projection of
    the normalized inputs X into the reduced dimensional space spanned by
    the first K columns of U. It returns the projected examples in Z.
    """

    Z = np.zeros((X.shape[0], K))
    Ureduce = U[:, :K]

    Z = np.matmul(X, Ureduce)

    return Z


def recoverData(Z, U, K):
    """RECOVERDATA Recovers an approximation of the original data when using the projected data
    X_rec = RECOVERDATA(Z, U, K) recovers an approximation the
    original data that has been reduced to K dimensions. It returns the
    approximate reconstruction in X_rec.
    """

    Xrec = np.zeros((Z.shape[0], U.shape[0]))

    Ureduce = U[:, :K]
    Xrecovered = np.matmul(Z, Ureduce.T)

    return Xrecovered


def displayData(X, **kwargs):
    """Uses module ex3helper's displayData to display data"""
    helper3.displayData(X, **kwargs)
