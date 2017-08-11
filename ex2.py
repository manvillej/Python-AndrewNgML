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
x = np.array(data[:,0:2]).transpose()


## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with \'+\' indicating (y = 1) examples and \'o\' indicating (y = 0) examples.')
helper.plotData(x,y)