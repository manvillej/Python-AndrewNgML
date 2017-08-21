## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid - complete
#     costFunction - complete
#     predict - complete
#     costFunctionReg - incomplete
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
import matplotlib.pyplot as plt
data = np.genfromtxt('./data/ex2data1.txt', delimiter=',')
y = np.array(data[:,2])
x = np.array(data[:,0:2])


## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.
[m,n] = x.shape

r = x
x = np.ones((m, n+1))
x[:,1:] = r

print('\nPlotting data with \'o\' indicating (y = 1) examples and \'x\' indicating (y = 0) examples.')

helper.plotData(x,y,'Exam Score 1', 'Exam Score 2')


input('\nPart 1 completed. Program paused. Press enter to continue: ')
## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m
#
#  Setup the data matrix appropriately, and add ones for the intercept term


theta = np.zeros(n+1)

cost = helper.costFunction(theta,x,y)
grad = helper.gradient(theta, x, y)


print('Cost at initial theta (zeros): {0:.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): ')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')


# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost = helper.costFunction(test_theta, x, y)
grad = helper.gradient(test_theta, x, y)

print('Cost at initial theta (zeros): {0:.3f}'.format(cost))
print('Expected cost (approx): 0.218')
print('Gradient at initial theta (zeros): ')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')

input('\nPart 2 completed. Program paused. Press enter to continue: ')

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc

results = helper.optimize(theta,x,y)
theta = results.x
cost = results.fun

# Print theta to screen
print('Cost at theta found by scipy.optimize.minimize with TNC: {0:.3f}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta:')
print(theta)
print('Expected theta (approx):')
print('[ -25.161  0.206  0.201]')
helper.plotDecisionBoundary(theta,x,y,'Exam Score 1', 'Exam Score 2')

input('\nPart 3 completed. Program paused. Press enter to continue: ')


## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m
#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = helper.sigmoid(np.matmul(np.array([1, 45, 85]), theta))
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)
print('Expected value: 0.775 +/- 0.002');

# Compute accuracy on our training set
p = helper.predict(theta, x)
predictions = np.zeros(p.shape)
predictions[np.where(p==y)] = 1


print('Train Accuracy: ', np.mean(predictions) * 100)
print('Expected accuracy (approx): 89.0\n')