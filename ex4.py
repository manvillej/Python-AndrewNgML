## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import ex3helper as helper3
import ex4helper as helper
import ex4Checker as checker

## Setup the parameters you will use for this exercise
inputLayerSize  = 400;  # 20x20 Input Images of Digits
hiddenLayerSize = 25;   # 25 hidden units
numLabels = 10;         # 10 labels, from 1 to 10   
                        # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')
mat = io.loadmat('./data/ex4data1.mat')
X = mat['X']
y = np.squeeze(mat['y'])

m = X.shape[0]
# Randomly select 100 data points to display
perm = np.random.permutation(m)
sel = X[perm[0:100],:]

helper3.displayData(sel)

input('\nPart 1 completed. Program paused. Press enter to continue: ')


## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...')

mat = io.loadmat('./data/ex4weights.mat')

theta1 = mat['Theta1']
theta2 = mat['Theta2']

nnParams = np.append(theta1.flatten(), theta2.flatten())


## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
lambdaVal = 0

J = helper.nnCostFunction(nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal)

print('Cost at parameters (loaded from ex4weights): {:.6f}'.format(J))
print('this value should be approx: 0.287629')

input('\nPart 2 & 3 completed. Program paused. Press enter to continue: ')

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... ')

# Weight regularization parameter (we set this to 1 here).
lambdaVal = 1

J = helper.nnCostFunction(nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal)

print('Cost at parameters (loaded from ex4weights): {:.6f}'.format(J))
print('this value should be approx: 0.383770')

input('\nPart 4 completed. Program paused. Press enter to continue: ')


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...')

g = helper.sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:');
print(g);

input('\nPart 5 completed. Program paused. Press enter to continue: ')

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...')

initialTheta1 = np.random.rand(inputLayerSize + 1, hiddenLayerSize)
initialTheta2 = np.random.rand(hiddenLayerSize + 1, numLabels)

# Unroll parameters
initialNNParams = np.append(initialTheta1.flatten(), initialTheta2.flatten())

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation... ')

#Check gradients by running checkNNGradients
checker.checkNNGradients(0)

input('\nPart 6 & 7 completed. Program paused. Press enter to continue: ')

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... ')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.


#  You should also try different values of lambda
lambdaVal = 3
checker.checkNNGradients(lambdaVal)

debug_J  = helper.nnCostFunction(nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal)

print('\nCost at parameters (loaded from ex4weights): {:.6f}'.format(debug_J))
print('this value should be approx: 0.576051')

input('\nPart 8 completed. Program paused. Press enter to continue: ')


## =================== Part 9: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#

print('\nTraining Neural Network... ')

MaxIter = 5000
lambdaVal = 1

results = helper.optimizeNN(initialNNParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, MaxIter)

finalThetaParams = results.x

input('\nPart 9 completed. Program paused. Press enter to continue: ')

## ================= Part 10: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... ')
finalTheta1 = finalThetaParams[:theta1.size]
finalTheta1.shape = theta1.shape
helper3.displayData(finalTheta1[:,1:])

input('\nPart 10 completed. Program paused. Press enter to continue: ')

## ================= Part 11: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

accuracy = helper.nnAccuracy(finalThetaParams, X, inputLayerSize, hiddenLayerSize, numLabels, y)

print('Training Set Accuracy: {:.2f}%'.format(accuracy))

input('\nPart 11 completed. Program complete. Press enter to exit: ')