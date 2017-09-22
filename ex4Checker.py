import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import ex4helper as helper
import math
import matplotlib.image as mpimg
from numpy import linalg as LA

def backPropMask(nnParams,*args):
	return helper.BackPropagation(nnParams,*args)
	
def costMask(nnParams,*args):
	return helper.nnCostFunction(nnParams,*args)

def checkNNGradients(lambdaVal):
	#   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
	#   backpropagation gradients, it will output the analytical gradients
	#   produced by your backprop code and the numerical gradients (computed
	#   using computeNumericalGradient). These two gradient computations should
	#   result in very similar values.
	#

	inputLayerSize = 3
	hiddenLayerSize = 5
	numLabels = 3
	m = 5

	#We generate some 'random' test data
	theta1 = debugInitializeWeights(hiddenLayerSize, inputLayerSize)
	theta2 = debugInitializeWeights(numLabels, hiddenLayerSize)
	
	# Reusing debugInitializeWeights to generate X
	X = debugInitializeWeights(m, inputLayerSize - 1); 
	y = np.remainder(np.arange(m),numLabels) + 1

	#unroll parameters
	nnParams = np.append(theta1.flatten(), theta2.flatten())

	grad = helper.BackPropagation(nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal)

	diff = op.check_grad(costMask, backPropMask, nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, epsilon=.0001)
	
	numGrad = op.approx_fprime(nnParams, costMask, .001 , inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal)
	# Visually examine the two gradient computations.  The two columns you get should be very similar.


	print('\nComparing Gradients: (numGrad, grad, absolute difference)')

	for i in range(0,numGrad.shape[0]):
		print("{}: {:.9f}, {:.9f} {:.9f}".format(i+1, numGrad[i], grad[i], abs(numGrad[i] - grad[i] )))
	

	print('The above two columns you get should be very similar.')
	print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')

	# Evaluate the norm of the difference between two solutions.  
	# If you have a correct implementation, and assuming you used EPSILON = 0.0001 
	# in computeNumericalGradient.m, then diff below should be less than 1e-9

	print('If your backpropagation implementation is correct, then ')
	print('the relative difference will be small (less than 1e-9).')
	print('Relative Difference: {}'.format(diff))


def debugInitializeWeights(fanOut, fanIn):
   	# Initialize W using "sin", this ensures that  vW is always of the same
	# values and will be useful for debugging
		# W = zeros(fan_out, 1 + fan_in);
		# W = reshape(sin(1:numel(W)), size(W)) / 10;
	# numel ~ number of elements. equivalent to size, w.size
	# size, equivalent of shape, w.shape
	W = np.arange(fanOut*(fanIn+1))
	W = W.reshape(fanOut, fanIn+1)
	W = np.sin(W)/10
	return W

if __name__ == '__main__':
	checkNNGradients(0)

