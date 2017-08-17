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
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
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
initial_theta = np.zeros([x.shape[1], 1])

# Set regularization parameter lambda to 1
lambdaVal = 1