"""Machine Learning Online Class
Exercise 5 | Regularized Linear Regression and Bias-Variance
Instructions
------------
This file contains code that helps you get started on the
exercise. You will need to complete the following functions:
   linearRegCostFunction - completed
   learningCurve - completed
   validationCurve - completed
For this exercise, you will not need to change any code in this file,
or any other files other than those mentioned above.
"""

# Imports:
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import ex5helper as helper


def main():
    # =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset.
    #  The following code will load the dataset into your environment and plot
    #  the data.

    print('Loading and Visualizing Data ...')

    # Load from ex5data1:
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    mat = io.loadmat('./data/ex5data1.mat')

    X = mat['X'].transpose()
    y = np.squeeze(mat['y'])

    Xval = mat['Xval'].transpose()
    yval = np.squeeze(mat['yval'])

    Xtest = mat['Xtest'].transpose()
    ytest = np.squeeze(mat['ytest'])

    # m = Number of examples
    m = X.shape[1]

    # Plot training data
    plt.scatter(X, y, marker='o', color='b', s=10)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()

    input('\nPart 1 completed. Program paused. Press enter to continue: ')

    #  =========== Part 2: Regularized Linear Regression Cost =============
    #  You should now implement the cost function for regularized linear
    #  regression.
    #
    theta = np.array([1, 1])

    lambdaVal = 1

    J = helper.linearRegressionCost(theta, X, y, lambdaVal)

    print('\nCost at theta = [1, 1]: {:.6f}'.format(J))
    print('(this value should be about 303.993192)')

    input('\nPart 2 completed. Program paused. Press enter to continue: ')

    #  =========== Part 3: Regularized Linear Regression Gradient =============
    #  You should now implement the gradient for regularized linear
    #  regression.

    grad = helper.linearRegressionGradient(theta, X, y, lambdaVal)
    print('Gradient at theta = [1, 1]:  [{:.6f}, {:.6f}]'.format(
        grad[0],
        grad[1]))
    print('(this value should be about: [-15.303016, 598.250744])')

    input('\nPart 3 completed. Program paused. Press enter to continue: ')

    #  =========== Part 4: Train Linear Regression =============
    #  Once you have implemented the cost and gradient correctly, the
    #  trainLinearReg function will use your cost function to train
    #  regularized linear regression.
    #  Write Up Note: The data is non-linear, so this will not give a great
    #                 fit.

    #  Train linear regression with lambda = 0
    lambdaVal = 0

    results = helper.trainLinearRegressionModel(theta, X, y, lambdaVal)
    theta = results.x

    # Plot training data
    plt.scatter(X, y, marker='o', color='b', s=10)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(
        X[0, :],
        helper.linearRegressionPredict(
            X,
            theta,
            addBias=True),
        color='red',
        linestyle='solid')
    plt.show()

    input('\nPart 4 completed. Program paused. Press enter to continue: ')

    #  =========== Part 5: Learning Curve for Linear Regression =============
    #  Next, you should implement the learningCurve function.
    #  Write Up Note: Since the model is underfitting the data, we expect to
    #                 see a graph with "high bias" -- Figure 3 in ex5.pdf

    [errorTrain, errorValidation] = helper.learningCurve(
        X,
        y,
        Xval,
        yval,
        lambdaVal)

    plt.plot(
        range(m),
        errorTrain,
        label="Training Error",
        color='blue',
        linestyle='solid')

    plt.plot(
        range(m),
        errorValidation,
        label="Cross Validation Error",
        color='red',
        linestyle='solid')

    plt.legend(loc='upper right')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Learning Curve for Linear Regression')
    plt.show()

    print('\n# of Training Examples: Train Error, Cross Validation Error')
    print('--------------------------------------------------------')
    for i in range(m):
        print('{:02d}: {:06.3f}, {:07.3f}'.format(
            i+1,
            errorTrain[i],
            errorValidation[i]))

    input('\nPart 5 completed. Program paused. Press enter to continue: ')

    #  ========= Part 6: Feature Mapping for Polynomial Regression ===========
    #  One solution to this is to use polynomial regression. You should now
    #  complete polyFeatures to map each example into its powers

    p = 8

    # Map X onto Polynomial Features and Normalize
    Xpoly = helper.polyFeatures(X, p)
    Xpoly = helper.featureNormalize(Xpoly)

    # Map X onto Polynomial Features and Normalize
    XpolyVal = helper.polyFeatures(Xval, p)
    XpolyVal = helper.featureNormalize(XpolyVal)

    # Map X onto Polynomial Features and Normalize
    XpolyTest = helper.polyFeatures(Xtest, p)
    XpolyTest = helper.featureNormalize(XpolyTest)

    print('Normalized Training Example 1:')
    print(Xpoly[:, 0])

    input('\nPart 6 completed. Program paused. Press enter to continue: ')

    #  ========== Part 7: Learning Curve for Polynomial Regression ============
    #  Now, you will get to experiment with polynomial regression with multiple
    #  values of lambda. The code below runs polynomial regression with
    #  lambda = 0. You should try running the code with different values of
    #  lambda to see how the fit and learning curve change.

    theta = np.ones(Xpoly.shape[0]+1)
    test = helper.linearRegressionGradient(theta, Xpoly, y, lambdaVal)

    plt.figure(1)
    plt.scatter(
        X,
        y,
        marker='o',
        color='b',
        s=10)

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = {})'.format(lambdaVal))
    helper.plotFit(
        X.min(),
        X.max(),
        X,
        theta,
        p)

    plt.figure(2)
    [errorTrain, errorValidation] = helper.learningCurve(
        Xpoly,
        y,
        XpolyVal,
        yval,
        lambdaVal)

    plt.plot(
        range(m),
        errorTrain,
        label="Training Error",
        color='blue',
        linestyle='solid')

    plt.plot(
        range(m),
        errorValidation,
        label="Cross Validation Error",
        color='red',
        linestyle='solid')

    plt.legend(loc='upper right')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(
        lambdaVal))

    print('\n# of Training Examples: Train Error, Cross Validation Error')
    print('--------------------------------------------------------')
    for i in range(m):
        print('{:02d}: {:06.3f}, {:07.3f}'.format(
            i+1,
            errorTrain[i],
            errorValidation[i]))

    plt.show()

    input('\nPart 7 completed. Program paused. Press enter to continue: ')

    #  You will now implement validationCurve to test various values of
    #  lambda on a validation set. You will then use this to select the
    #  "best" lambda value.

    lambdaVector = np.array([0, .001, .003, .01, .03, .1, .3, 1, 3, 10])

    [errorTrain, errorValidation] = helper.validationCurve(
        Xpoly,
        y,
        XpolyVal,
        yval,
        lambdaVector)

    print('\nRow #, Lambda: Train Error, Cross Validation Error')
    print('--------------------------------------------------------')
    for i in range(lambdaVector.size):
        print('{:02d}. {:05.3f}: {:06.3f}, {:07.3f}'.format(
            i+1,
            lambdaVector[i],
            errorTrain[i],
            errorValidation[i]))

    position = np.arange(lambdaVector.size)
    barWidth = .25
    plt.bar(
        position - barWidth/2,
        errorTrain,
        barWidth,
        label="Training Error",
        color='blue')

    plt.bar(
        position + barWidth/2,
        errorValidation,
        barWidth,
        label="Cross Validation Error",
        color='red')

    plt.xticks(position, lambdaVector)
    plt.legend(loc='upper left')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.title('Polynomial Regression Validation Curve')
    plt.show()

    input('\nPart 8 completed. Program completed. Press enter to exit: ')

if __name__ == '__main__':
    main()
