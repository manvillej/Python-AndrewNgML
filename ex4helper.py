import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as op
import math
import matplotlib.image as mpimg

def nnCostFunction(nnParams, inputSize, hiddenLayerSize, outputSize, X, y, lambdaVal):
	#get num examples
	m = X.shape[0]

	#get Theta Matrices
	[theta1, theta2] = getThetas(nnParams,inputSize,hiddenLayerSize,outputSize)

	#prepare Y matrix for cost function
	Y = getYMatrix(y)

	#forward Pass
	[a1, z1, a2, z2, h2] = forwardPass(np.array([theta1, theta2]), X)


	#getting regulation parameters
	R1 = theta1[:,1:]
	R2 = theta2[:,1:]

	# calculating the cost of regulation
	costRegulation = lambdaVal*(np.sum(np.square(R1.flatten())) + np.sum(np.square(R2.flatten())))/(2*m)
	
	#calculating true cost without regulation
	cost = np.sum(np.log(h2)*Y) + np.sum(np.log(1-h2)*(1-Y))
	cost = -cost/m

	#calculate total cost
	totalCost = cost + costRegulation

	return totalCost

def BackPropagation(nnParams, inputSize, hiddenLayerSize, outputSize, X, y, lambdaVal):
	#get num examples
	m = X.shape[0]
	#get Theta Matrices
	[theta1, theta2] = getThetas(nnParams,inputSize,hiddenLayerSize,outputSize)


	#prepare Y matrix for cost function
	Y = getYMatrix(y) #5x3

	#forward Pass
	[a1, z1, a2, z2, h2] = forwardPass(np.array([theta1, theta2]), X)
	#a1 = 5x4, z1 = 5x5, a2 = 5x5, a2 = 5x6, z2 = 5x3, h2 = 5x3

	#backward
	theta2Error = h2-Y #5x3
	theta1Error = np.matmul(theta2Error,theta2[:,1:])*sigmoidGradient(z1)
	
	D1 = np.matmul(theta1Error.transpose(),a1)
	D2 = np.matmul(theta2Error.transpose(),a2)

	#average the gradient per example	
	theta1Grad = D1/m
	theta2Grad = D2/m

	#calculate regulation terms
	theta1Reg = lambdaVal*theta1/m
	theta2Reg = lambdaVal*theta2/m
	theta1Reg[:,0] = 0
	theta2Reg[:,0] = 0

	#combine gradient and regulation terms	
	theta1Grad = theta1Grad + theta1Reg
	theta2Grad = theta2Grad + theta2Reg

	return np.append(theta1Grad.flatten(), theta2Grad.flatten())

def forwardPass(nnParams, X):
	theta1 = nnParams[0]
	theta2 = nnParams[1]

	#left side is the example count
	#layer 1
	a1 = np.insert(X,0,np.ones(X.shape[0]),axis=1)#5x4
	z1 = np.matmul(a1,theta1.transpose())#5x5
	a2 = sigmoid(z1)#5x5


	#layer 2
	a2 = np.insert(a2,0,np.ones(a1.shape[0]),axis=1) # adding bias unit  5x6
	z2 = np.matmul(a2,theta2.transpose()) #5x3
	a3 = sigmoid(z2) #5x3

	return [a1, z1, a2, z2, a3]

def getYMatrix(y):
	#prepare Y matrix for cost function
	numLabels = np.unique(y).shape[0]

	#create boolean array of value or not out of 1s and 0s
	Y = (y==1).astype(int)
	for i in range(2, numLabels + 1):
		Y = np.append(Y,(y==i).astype(int))
	#reshape so first dimension corresponds with label
	Y = Y.reshape(numLabels,y.shape[0])
	return Y.transpose()

def getThetas(nnParams,inputSize,hiddenLayerSize,outputSize):
	theta1Length = (inputSize+1)*hiddenLayerSize

	theta1 = nnParams[:theta1Length]
	theta2 = nnParams[theta1Length:]

	theta1 = theta1.reshape(hiddenLayerSize,inputSize+1)
	theta2 = theta2.reshape(outputSize,hiddenLayerSize+1)

	return[theta1, theta2]

def sigmoidGradient(Z):
	R = sigmoid(Z)
	return R*(1-R)

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def optimizeNN(nnParams, inputSize, hiddenLayerSize, outputSize, X, y, lambdaVal, maxIter):
	return op.minimize(fun=nnCostFunction, x0=nnParams, args=(inputSize, hiddenLayerSize, outputSize, X, y, lambdaVal), method='TNC', jac = BackPropagation, options={'maxiter': maxIter, 'disp': True})
