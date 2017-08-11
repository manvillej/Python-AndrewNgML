## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid.m - incomplete
#     costFunction.m - incomplete
#     predict.m - incomplete
#     costFunctionReg.m - incomplete
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
import numpy as np
import ex2helper as helper

data = np.genfromtxt('./data/ex2data1.txt', delimiter=',')
y = np.array(data[:,2:])
x = np.array(data[:,0:2])


## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('\nPlotting data with \'o\' indicating (y = 1) examples and \'x\' indicating (y = 0) examples.')

helper.plotData(x,y)

input('\nPart 1 completed. Program paused. Press enter to continue: ')
## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m
#
#  Setup the data matrix appropriately, and add ones for the intercept term
[m,n] = x.shape

r = x
x = np.ones((m, n+1))
x[:,1:] = r

theta = np.zeros(n+1)

[cost, grad] = helper.costFunction(theta,x,y)


print('Cost at initial theta (zeros): {0:.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): ')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')


# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
[cost, grad] = helper.costFunction(test_theta, x, y)

print('Cost at initial theta (zeros): {0:.3f}'.format(cost))
print('Expected cost (approx): 0.218')
print('Gradient at initial theta (zeros): ')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')

input('\nPart 2 completed. Program paused. Press enter to continue: ')