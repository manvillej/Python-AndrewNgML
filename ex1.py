## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This is the initialization file for exercise 1 of Andrew Ng's Machine learning course
#  All of the file have been converted into a python implementation instead of the original
#  Matlab implementation. This 
#
#     warmUpExercise.py - complete
#     plotData.py - complete
#     gradientDescent.py - in progress
#     computeCost.py - incomplete
#     gradientDescentMulti.py - incomplete
#     computeCostMulti.py - incomplete
#     featureNormalize.py - incomplete
#     normalEqn.py - incomplete
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#
# 
## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
import numpy as np
import matplotlib.pyplot as plt

print("running warmUpExercise...")
print('5x5 Identity Matrix:')

eye = np.identity(5)
print(eye)

input('\nPart 1 completed. Program paused. Press enter to continue: ')

## ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.genfromtxt('ex1data1.txt', delimiter=',')

x=data[:,0]
y=data[:,1]

plt.scatter(x, y, label = "scatter", color='r', s=10)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Raw Data')
plt.show()
input('\nPart 2 completed. Program paused. Press enter to continue: ')

## =================== Part 3: Cost and Gradient descent ===================

