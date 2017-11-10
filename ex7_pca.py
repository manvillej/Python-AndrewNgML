""" Machine Learning Online Class Exercise 7 | Principle Component Analysis and K-Means Clustering
Instructions
------------
This file contains code that helps you get started on the
exercise. You will need to complete the following functions:
   pca
   projectData
   recoverData
   computeCentroids
   findClosestCentroids
   kMeansInitCentroids
For this exercise, you will not need to change any code in this file,
or any other files other than those mentioned above.
"""

# Imports
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import ex7helper as helper
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


def main():
    #  ================== Part 1: Load Example Dataset  ===================
    #  We start this exercise by using a small dataset that is easily to
    #  visualize
    print('\nVisualizing example dataset for PCA...\n')

    #  The following command loads the dataset. You should now have the
    #  variable X in your environment
    mat = io.loadmat('./data/ex7data1.mat')
    X = mat['X']

    plt.title('Raw Data')
    plt.scatter(
        X[:, 0],
        X[:, 1],
        marker='o',
        color='b',
        s=10)

    plt.show()

    input('Part 1 completed. Program paused. Press enter to continue: ')

    #  =============== Part 2: Principal Component Analysis ===============
    #  You should now implement PCA, a dimension reduction technique. You
    #  should complete the code in pca
    print('Running PCA on example dataset...\n')

    #  Before running PCA, it is important to first normalize X
    Xnorm = helper.featureNormalize(X)

    #  Run PCA
    [U, S, V] = helper.PCA(Xnorm)

    #  Compute mu, the mean of the each feature
    mu = np.mean(X, axis=0)

    #  Draw the eigenvectors centered at mean of data. These lines show the
    #  directions of maximum variations in the dataset.
    plt.title('Data plotted with PCA vectors')
    plt.scatter(
        X[:, 0],
        X[:, 1],
        marker='o',
        color='b',
        s=10)

    helper.drawLines(
        mu,
        mu + 1.5*S[0]*U[0, :],
        'r')

    helper.drawLines(
        mu,
        mu + 1.5*S[1]*U[1, :],
        'g')

    plt.show()

    print('Top eigenvector: ')
    print(U[0, :])
    print('(you should expect to see -0.707107 -0.707107)\n')

    input('Part 2 completed. Program paused. Press enter to continue: ')

    #  =================== Part 3: Dimension Reduction ===================
    #  You should now implement the projection step to map the data onto the
    #  first k eigenvectors. The code will then plot the data in this reduced
    #  dimensional space.  This will show you what the data looks like when
    #  using only the corresponding eigenvectors to reconstruct it.

    print('Dimension reduction on example dataset...\n')

    #  Plot the normalized dataset (returned from pca)
    plt.scatter(
        Xnorm[:, 0],
        Xnorm[:, 1],
        marker='o',
        color='b',
        s=10)

    plt.axis([-4, 3, -4, 3])  # axis square

    #  Project the data onto K = 1 dimension
    K = 1
    Z = helper.projectData(Xnorm, U, K)

    print('Projection of the first example: {}'.format(Z[0]))
    print('(this value should be about 1.481274)\n')

    Xrecovered = helper.recoverData(Z, U, K)

    print('Approximation of the first example: {:.6f} {:.6f}'.format(
        Xrecovered[0, 0],
        Xrecovered[0, 1]))

    print('(this value should be about  -1.047419 -1.047419)\n')

    plt.title('Data plotted with PCA vectors')
    mappedX = 1.5*np.array(
        [np.max(
            Xrecovered[:, 0]),
            np.min(Xrecovered[:, 0])])
    mappedY = 1.5*np.array(
        [np.max(
            Xrecovered[:, 1]),
            np.min(Xrecovered[:, 1])])

    plt.plot(
        mappedX,
        mappedY,
        color='g',
        linewidth=1)

    for i in range(Xnorm.shape[0]):
        helper.drawLines(
            Xnorm[i, :],
            Xrecovered[i, :],
            'k')

    plt.show()

    input('Part 3 completed. Program paused. Press enter to continue: ')

    #  =============== Part 4: Loading and Visualizing Face Data =============
    #  We start the exercise by first loading and visualizing the dataset.
    #  The following code will load the dataset into your environment
    print('Loading face dataset.\n')

    mat = io.loadmat('./data/ex7faces.mat')
    X = mat['X']

    #  Display the first 100 faces in the dataset
    helper.displayData(X[:100, :])
    plt.show()

    input('Part 4 completed. Program paused. Press enter to continue: ')

    #  =========== Part 5: PCA on Face Data: Eigenfaces  ===================
    #  Run PCA and visualize the eigenvectors which are in this case eigenfaces
    #  We display the first 36 eigenfaces.
    print('Running PCA on face dataset. (this might take a minute or two ...)\n')

    #  Before running PCA, it is important to first normalize X by subtracting
    #  the mean value from each feature

    Xnorm = helper.featureNormalize(X)

    #  Run PCA
    [U, S, V] = helper.PCA(Xnorm)

    #  Visualize the top 36 eigenvectors found
    helper.displayData(U[:, :36].T)
    plt.show()

    input('Part 5 completed. Program paused. Press enter to continue: ')

    #  ============= Part 6: Dimension Reduction for Faces =================
    #  Project images to the eigen space using the top k eigenvectors
    #  If you are applying a machine learning algorithm
    print('Dimension reduction for face dataset...')

    K = 100
    Z = helper.projectData(Xnorm, U, K)
    print('The projected data Z has a size of: ', Z.shape)

    input('\nPart 6 completed. Program paused. Press enter to continue: ')

    #  ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
    #  Project images to the eigen space using the top K eigen vectors and
    #  visualize only using those K dimensions
    #  Compare to the original input, which is also displayed
    print('Visualizing the projected (reduced dimension) faces....\n')

    Xrecovered = helper.recoverData(Z, U, K)

    # Display normalized data

    # Charting
    plt.figure(2)
    plt.title('Original Data')
    helper.displayData(Xnorm[:100, :])

    plt.figure(1)
    plt.title('Recovered Data')
    helper.displayData(Xrecovered[:100, :])

    # Display
    plt.show()

    input('Part 7 completed. Program paused. Press enter to continue: ')

    #  === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
    #  One useful application of PCA is to use it to visualize high-dimensional
    #  data. In the last K-Means exercise you ran K-Means on 3-dimensional
    #  pixel colors of an image. We first visualize this output in 3D, and then
    #  apply PCA to obtain a visualization in 2D.
    #  Reload the image from the previous exercise and run K-Means on it
    #  For this to work, you need to complete the K-Means assignment first
    print('Visualize groupings in 3d...')

    X = plt.imread('./data/bird_small.png')
    A = X/255  # divide by 255 to reduce values to a range of 0 - 1

    # image shape
    imgShape = A.shape

    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    newShape = [imgShape[0]*imgShape[1], imgShape[2]]

    A = np.reshape(A, newShape)

    # Run your K-Means algorithm on this data
    # You should try different values of K and max_iters here
    K = 16

    # Calculate Kmeans clusters
    kmeans = KMeans(
        n_clusters=K,
        random_state=0).fit(A)

    centroids = kmeans.cluster_centers_
    idx = kmeans.labels_

    #  Sample 1000 random indexes (since working with all the data is
    #  too expensive. If you have a fast computer, you may increase this.
    perm = np.random.permutation(A.shape[0])
    perm = perm[:1000]

    #  Setup Color Palette
    # palette = hsv(K);
    # colors = palette(idx(sel), :);
    #  Visualize the data and centroid memberships in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        A[perm, 0],
        A[perm, 1],
        zs=A[perm, 2],
        c=idx[perm],
        cmap=plt.get_cmap('hsv'))

    plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    input('\nPart 8A completed. Program paused. Press enter to continue: ')

    #  === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
    # Use PCA to project this cloud to 2D for visualization
    print('Visualize groupings in 2d...')

    # Subtract the mean to use PCA
    Anorm = helper.featureNormalize(A)

    #  PCA and project the data to 2D
    #  Run PCA
    [U, S, V] = helper.PCA(Anorm)
    Z = helper.projectData(Anorm, U, K)

    # Plot in 2D
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        Z[perm, 0],
        Z[perm, 1],
        c=idx[perm],
        cmap=plt.get_cmap('hsv'))

    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    plt.show()

    input('\nPart 8b completed. Program completed. Press enter to exit: ')

if __name__ == '__main__':
    main()
