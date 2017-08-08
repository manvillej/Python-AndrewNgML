## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  You will need to complete the following functions in this 
#  exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#
#
## Initialization
#
## ================ Part 1: Feature Normalization ================
#
## Clear and Close Figures
import numpy as np

print('Loading data...')
data = np.genfromtxt('ex1data2.txt', delimiter=',')
x = np.array(data[:,:2])
y = np.array(data[:,2])
print(x.shape)
print(y.shape)