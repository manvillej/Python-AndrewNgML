"""Machine Learning Online Class
Exercise 6 | Support Vector Machines
Instructions
------------
This file contains code that helps you get started on the
exercise. You will need to complete the following functions:
   gaussianKernel
   dataset3Params
   processEmail
   emailFeatures
For this exercise, you will not need to change any code in this file,
or any other files other than those mentioned above.
"""

# Imports:
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from sklearn import svm
import ex6helper as helper


def main():
    #  =============== Part 1: Loading and Visualizing Data ================
    #  We start the exercise by first loading and visualizing the dataset.
    #  The following code will load the dataset into your environment and plot
    #  the data.
    print('Loading and Visualizing Data ...')

    # Load from ex6data1:
    # You will have X, y in your environment

    mat = io.loadmat('./data/ex6data1.mat')

    X = mat['X']

    y = mat['y'].astype(int).ravel()

    helper.plotData(X, y, addBias=True)
    plt.show()

    input('\nPart 1 completed. Program paused. Press enter to continue: ')

    #  ==================== Part 2: Training Linear SVM ====================
    #  The following code will train a linear SVM on the dataset and plot the
    #  decision boundary learned.
    print('\nTraining Linear SVM ...')

    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    C = 1

    model = svm.SVC(
        C=1,
        max_iter=100,
        tol=.01,
        kernel='linear')

    model.fit(X, y)
    helper.visualizeBoundary(X, y, model)
    plt.show()

    input('\nPart 2 completed. Program paused. Press enter to continue: ')

    # =============== Part 3: Implementing Gaussian Kernel ===============
    #  You will now implement the Gaussian kernel to use
    #  with the SVM. You should complete the code in gaussianKernel

    print('\nEvaluating the Gaussian Kernel ...')

    X1 = np.array([1, 2, 1])
    X2 = np.array([0, 4, -1])
    sim = helper.gaussianKernel(X1, X2, sigma=2)

    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {:.6f}'.format(sim))
    print('(for sigma = 2, this value should be about 0.324652)')

    input('\nPart 3 completed. Program paused. Press enter to continue: ')

    #  =============== Part 4: Visualizing Dataset 2 ================
    #  The following code will load the next dataset into your environment and
    #  plot the data.

    print('\nLoading and Visualizing Data ...')

    mat = io.loadmat('./data/ex6data2.mat')

    X = mat['X']
    y = mat['y'].astype(int).ravel()

    helper.plotData(X, y, addBias=True)
    plt.show()

    input('\nPart 4 completed. Program paused. Press enter to continue: ')

    #  ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
    #  After you have implemented the kernel, we can now use it to train the
    #  SVM classifier.
    print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

    # Train the SVM with the Gaussian kernel on this dataset.
    sigma = 0.1
    gamma = np.power(sigma, -2.)

    model = svm.SVC(
        C=1,
        kernel='rbf',
        gamma=gamma)

    model.fit(X, y.flatten())
    helper.visualizeBoundary(X, y, model)
    plt.show()

    input('\nPart 5 completed. Program paused. Press enter to continue: ')

    #  =============== Part 6: Visualizing Dataset 3 ================
    #  The following code will load the next dataset into your environment and
    #  plot the data.

    print('\nLoading and Visualizing Data ...')

    mat = io.loadmat('./data/ex6data3.mat')

    X = mat['X']
    y = mat['y'].astype(int).ravel()

    helper.plotData(X, y, addBias=True)
    plt.show()

    input('\nPart 6 completed. Program paused. Press enter to continue: ')

    #  ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
    #  This is a different dataset that you can use to experiment with. Try
    #  different values of C and sigma here.
    #

    Xval = mat['Xval']
    yval = mat['yval'].astype(int).ravel()

    # get optimal parameters
    [C, sigma] = helper.dataset3Params(
        X,
        y,
        Xval,
        yval)

    gamma = np.power(sigma, -2.)

    print('\nFound C & Sigma: {} & {}'.format(C, sigma))

    # train the model
    model = svm.SVC(
        C=C,
        kernel='rbf',
        gamma=gamma)

    model.fit(X, y.flatten())
    helper.visualizeBoundary(X, y, model)
    plt.show()

    input('\nPart 7 completed. Program completed. Press enter to exit: ')

if __name__ == '__main__':
    main()
