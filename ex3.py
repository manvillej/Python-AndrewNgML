## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function) - complete
#     oneVsAll.m - complete
#     predictOneVsAll.m - complete
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import ex2helper as helper2
import ex3helper as helper

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')
mat = io.loadmat('./data/ex3data1.mat')
X = mat['X']
y = np.squeeze(mat['y'])


m = X.shape[0]

# Randomly select 100 data points to display
perm = np.random.permutation(m)
sel = X[perm[0:100],:]

#display data as image
helper.displayData(sel)
 
input('\nPart 1 completed. Program paused. Press enter to continue: ')

## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.


# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2,-1,1,2])
X_t = np.concatenate((np.array([np.ones(5)]),np.divide(np.arange(1,16,1),10).reshape(3,5)),axis=0).transpose()
Y_t = np.array([1,0,1,0,1])
lambda_t = 3

J = helper2.costFunctionReg(theta_t,X_t,Y_t,lambda_t)
grad = helper2.gradientReg(theta_t,X_t,Y_t,lambda_t)

print('Cost: {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('Gradients:')
print(grad)
print('Expected gradients:')
print('[0.146561 -0.548558 0.724722 1.398003]')


input('\nPart 2a completed. Program paused. Press enter to continue: ')

## ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...')

lambdaVal = .1
allTheta = helper.OneVsAll(X, y, np.unique(y), lambdaVal)


input('\nPart 2b completed. Program paused. Press enter to continue: ')
## ================ Part 3: Predict for One-Vs-All ================

p = helper.predictOneVsAll(allTheta,X)
predictions = np.zeros(p.shape)
predictions[np.where(p==y)] = 1

print('Train Accuracy: {:.1f}%'.format(np.mean(predictions) * 100))
print('Expected Accuracy: 96.5%')
