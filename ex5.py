## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import ex5helper as helper


## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

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

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#
theta = np.array([1, 1])
X = np.insert(X,0,np.ones(X.shape[0]),axis=0)
Xval = np.insert(Xval,0,np.ones(Xval.shape[0]),axis=0)
Xtest = np.insert(Xtest,0,np.ones(Xtest.shape[0]),axis=0)

lambdaVal = 1

J = helper.linearRegressionCost(theta, X, y, lambdaVal)

print('\nCost at theta = [1, 1]: {:.6f}'.format(J))
print('(this value should be about 303.993192)')


input('\nPart 2 completed. Program paused. Press enter to continue: ')

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

grad = helper.linearRegressionGradient(theta, X, y, lambdaVal)
print('Gradient at theta = [1, 1]:  [{:.6f}, {:.6f}]'.format(grad[0],grad[1]))
print('(this value should be about: [-15.303016, 598.250744])')

input('\nPart 3 completed. Program paused. Press enter to continue: ')

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0
lambdaVal = 0

results = helper.trainLinearRegressionModel(theta, X, y, lambdaVal)
theta = results.x

# Plot training data
plt.scatter(X[1,:], y, marker='o', color='b', s=10)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X[1,:],helper.linearRegressionPredict(X, theta), color='red', linestyle='solid')
plt.show()

input('\nPart 4 completed. Program paused. Press enter to continue: ')

## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

[errorTrain, errorValidation] = helper.learningCurve(X,y,Xval,yval,lambdaVal)

trainingError = plt.plot(range(m),errorTrain,  label = "Training Error", color='blue', linestyle='solid')
crossValidationError = plt.plot(range(m), errorValidation, label = "Cross Validation Error", color='red', linestyle='solid')
plt.legend(loc='upper right')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Learning Curve for Linear Regression')
plt.show()

print('\n# of Training Examples: Train Error, Cross Validation Error')
print('--------------------------------------------------------')
for i in range(m):
	print('{:2d}: {:.3f}, {:.3f}'.format(i+1,errorTrain[i],errorValidation[i]))

input('\nPart 5 completed. Program paused. Press enter to continue: ')

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#