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
#     computeCost.py - complete
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
import ex1helper as helper
import matplotlib.pyplot as plt

print("running warmUpExercise...")
print('5x5 Identity Matrix:')

eye = np.identity(5)
print(eye)

input('\nPart 1 completed. Program paused. Press enter to continue: ')

## ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.genfromtxt('ex1data1.txt', delimiter=',')

x=np.array(data[:,0])
x=np.expand_dims(x,axis=0)
x=np.append(np.ones_like(x),x,axis=0)
y=np.array(data[:,1])

plt.scatter(x[1], y, label = "scatter", color='r', s=10)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Raw Data')
plt.show()
input('\nPart 2 completed. Program paused. Press enter to continue: ')

## =================== Part 3: Cost and Gradient descent ===================
theta = np.zeros(x.shape[0])

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

print('Testing the cost function ...')
# compute and display initial cost
J = helper.computeCost(x,y,theta)
print("With theta = [0 0], \nCost computed = ", J)
print("Expected cost value (approx) 32.07")


# further testing of the cost function
J = helper.computeCost(x, y, [-1, 2]);
print("With theta = [-1 2], \nCost computed = ", J)
print('Expected cost value (approx) 54.24');

input('\n Program paused. Press enter to continue: ')

print('Running Gradient Descent...')