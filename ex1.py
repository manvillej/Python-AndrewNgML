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
#     gradientDescent.py - complete
#     computeCost.py - complete
#     gradientDescentMulti.py - complete
#     computeCostMulti.py - complete
#     featureNormalize.py - complete
#     normalEqn.py - complete
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#
# 
## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
import numpy as np
import ex1helper as helper
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib


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

plt.scatter(x[1], y, label = "scatter", marker='x', color='r', s=10)
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
#run gradient descent
theta, cost = helper.gradientDescent(x, y, theta, alpha, iterations)

#print theta to screen
print('Theta found by gradient descent:');
print(theta)
print('\nExpected theta values (approx):');
print('[-3.6303 1.1664]');

# Plot the linear fit
plt.scatter(x[1], y, label = "scatter", marker='x', color='r', s=10)
plt.plot(x[1],np.matmul(x.transpose(),theta), color='blue', linestyle='solid')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Raw Data + Linear Fit')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.matmul([1, 3.5],theta) 
print('For population = 35,000, we predict a profit of ', predict1*10000);
predict2 = np.matmul([1, 7],theta)
print('For population = 70,000, we predict a profit of ', predict2*10000);

input('\nPart 3 completed. Program paused. Press enter to continue: ')

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')


#Grid over which we will calculate J

theta0 = np.linspace(-10, 10, 100)
theta1 = np.linspace(-1, 4, 100)
theta0Vals, theta1Vals = np.meshgrid(theta0,theta1)
zs = np.array([helper.computeCost(x,y,[i,j]) for i,j in zip(np.ravel(theta0Vals), np.ravel(theta1Vals))])
ZCosts = zs.reshape(theta0Vals.shape)

min = np.amin(ZCosts)
max = np.amax(ZCosts)
norm = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)



fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(theta0Vals,theta1Vals,ZCosts,cmap=cm.coolwarm, norm=norm)


ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('Cost')

plt.figure(2)
CS = plt.contour(theta0Vals, theta1Vals, ZCosts, np.logspace(-2,3,20))
plt.scatter(theta[0], theta[1], label = "scatter", marker='x', color='r', s=10)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.show()