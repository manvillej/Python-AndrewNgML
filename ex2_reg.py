## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m - complete
#     costFunction.m - complete
#     predict.m - complete
#     costFunctionReg.m - complete
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).
import numpy as np
import ex2helper as helper
import matplotlib.pyplot as plt

data = np.genfromtxt('./data/ex2data2.txt', delimiter=',')
y = np.array(data[:,2])
x = np.array(data[:,0:2])

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

x = helper.mapFeatures(x)

# Initialize fitting parameters
initial_theta = np.zeros(x.shape[1])

# Set regularization parameter lambda to 1
lambdaVal = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = helper.costFunctionReg(initial_theta, x, y, lambdaVal)
grad = helper.gradientReg(initial_theta, x, y, lambdaVal)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')
print('\nGradient at initial theta (zeros) - first five values only:')
print(" {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}".format(grad[0], grad[1],grad[2],grad[3],grad[4]))
print('Expected gradients (approx) - first five values only:')
print(' 0.0085 0.0188 0.0001 0.0503 0.0115\n')


# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(x.shape[1])
cost = helper.costFunctionReg(test_theta, x, y, 10)
grad = helper.gradientReg(test_theta, x, y, 10)


print('Cost at test theta (with lambda = 10): {:.2f}'.format(cost))
print('Expected cost (approx): 3.16')
print('\nGradient at initial theta (zeros) - first five values only:')
print(" {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}".format(grad[0], grad[1],grad[2],grad[3],grad[4]))
print('Expected gradients (approx) - first five values only:')
print(' 0.3460 0.1614 0.1948 0.2269 0.0922');

input('\nPart 1 completed. Program paused. Press enter to continue: ')


## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
theta = np.zeros(x.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lambdaVal = 1

results = helper.optimizeReg(theta,x,y,lambdaVal)
print(x.shape)
print(theta.shape)
print(y.shape)
theta = results.x
cost = results.fun

helper.plotData(x,y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('Raw Data')
plt.show()

helper.plotDecisionBoundary(theta,x,y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('Lambda = {:}'.format(lambdaVal))
plt.show()

# Compute accuracy on our training set
p = helper.predict(theta, x)
predictions = np.zeros(p.shape)
predictions[np.where(p==y)] = 1

print('Train Accuracy: {:.1f}'.format(np.mean(predictions) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')