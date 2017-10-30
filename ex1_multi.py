"""Machine Learning Online Class Exercise 1: Linear regression with multiple variables

Instructions
------------
This file contains code that helps you get started on the
linear regression exercise.
You will need to complete the following functions in this
exericse:
   warmUpExercise.py - complete
   plotData.py - complete
   gradientDescent.py - complete
   computeCost.py - complete
   gradientDescentMulti.py - complete
   computeCostMulti.py - complete
   featureNormalize.py - complete
   normalEqn.py - complete

For this part of the exercise, you will need to change some
parts of the code below for various experiments (e.g., changing
learning rates).
"""

# imports
import numpy as np
import ex1helper as helper
import matplotlib.pyplot as plt


def main():
    # ================ Part 1: Feature Normalization ================
    #
    # Clear and Close Figures

    print('Loading data...')
    data = np.genfromtxt('./data/ex1data2.txt', delimiter=',')
    x = np.array(data[:, :2])
    y = np.array(data[:, 2])
    m = y.shape[0]

    # Print out some data points
    print('First 10 examples from the dataset: ')
    for i in range(0, 10):
        print("x = [%.0f %.0f], y = %.0f" % (x[i, 0], x[i, 1], y[i]))

    input('Program paused. Press enter to continue: ')

    print('\nNormalize Features...')

    x, mu, sigma = helper.featureNormalize(x)

    # add bias unit
    r = x
    x = np.ones((x.shape[0], x.shape[1]+1))
    x[:, 1:] = r

    # ================ Part 2: Gradient Descent ================

    # ====================== YOUR CODE HERE ======================
    # Instructions: We have provided you with the following starter
    #               code that runs gradient descent with a particular
    #               learning rate (alpha).
    #
    #               Your task is to first make sure that your functions -
    #               computeCost and gradientDescent already work with
    #               this starter code and support multiple variables.
    #
    #               After that, try running gradient descent with
    #               different values of alpha and see which one gives
    #               you the best result.
    #
    #               Finally, you should complete the code at the end
    #               to predict the price of a 1650 sq-ft, 3 br house.
    #
    # Hint: By using the 'hold on' command, you can plot multiple
    #       graphs on the same figure.
    #
    # Hint: At prediction, make sure you do the same feature normalization.
    #
    print('\n\nPart 1 complete.')
    print('\nRunning gradient descent ...')

    # Choose some alpha value
    alpha = 0.01
    num_iters = 400

    # Init Theta and Run Gradient Descent
    theta = np.zeros(3)
    [theta, J_history] = helper.gradientDescentMulti(x, y, theta, alpha, num_iters)
    plt.plot(range(0, num_iters), J_history, color='blue', linestyle='solid')
    plt.xlabel('iterations')
    plt.ylabel('Cost J')
    plt.show()
    print('Theta computed from gradient descent:', theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================
    # Recall that the first column of X is all-ones. Thus, it does
    # not need to be normalized.

    r = np.array([1650, 3])
    r = (r - mu)/sigma
    r2 = np.ones(r.shape[0]+1)
    r2[1:] = r
    r = r2
    price = np.matmul(r, theta)

    # ============================================================
    print('\nPredicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${0:.2f}'.format(price))

    input('\nPart 2 complete. Program paused. Press enter to continue: ')

    # ================ Part 3: Normal Equations ================

    print('Solving with normal equations...')

    # ====================== YOUR CODE HERE ======================
    # Instructions: The following code computes the closed form
    #               solution for linear regression using the normal
    #               equations. You should complete the code in
    #               normalEqn.m
    #
    #               After doing so, you should complete this code
    #               to predict the price of a 1650 sq-ft, 3 br house.
    #

    # Load Data
    data = np.genfromtxt('./data/ex1data2.txt', delimiter=',')
    x = np.array(data[:, :2])
    y = np.array(data[:, 2])
    m = y.shape[0]

    # Add intercept term to X
    r = x
    x = np.ones((x.shape[0], x.shape[1]+1))
    x[:, 1:] = r

    # Calculate the parameters from the normal equation
    theta = helper.normalEqn(x, y)

    # Display normal equation's result
    print('Theta computed from the normal equations: ', theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================

    r = np.array([1, 1650, 3])
    price = np.matmul(r, theta)

    # ============================================================
    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${0:.2f}'.format(price))

if __name__ == '__main__':
    main()
