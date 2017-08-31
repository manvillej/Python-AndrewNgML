import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import ex2helper as helper
import math
import matplotlib.image as mpimg

def nnCostFunction(nnParams, X, y, lambdaVal):
	m = X.shape[0]
	print(m)
	theta1 = nnParams[0]
	n1 = X.shape[1] + 1
	n2 = theta1.shape[0]
	theta1 = theta1.reshape(int(n2/n1),n1)
	theta2 = nnParams[1]
	n1 = theta1.shape[0] + 1
	n2 = theta2.shape[0]
	theta2 = theta2.reshape(int(n2/n1),n1)
	

	#prepare Y matrix for cost function
	numLabels = np.unique(y).shape[0]+1
	#create boolean array of value or not out of 1s and 0s
	Y = (y==1).astype(int)
	for i in range(2,numLabels):
		Y = np.append(Y,(y==i).astype(int))
	#reshape so first dimension corresponds with label
	Y = Y.reshape(10,5000)

	X = np.insert(X,0,np.ones(X.shape[0]),axis=1) # adding bias unit
	h1 = helper.sigmoid(np.matmul(X,theta1.transpose()))
	h1 = np.insert(h1,0,np.ones(h1.shape[0]),axis=1) # adding bias unit
	h2 = helper.sigmoid(np.matmul(h1,theta2.transpose())).transpose()

	#getting regulation parameters
	R1 = theta1[:,1:]
	R2 = theta2[:,1:]
	# calculating the cost of regulation
	costRegulation = lambdaVal*(np.sum(np.square(R1.flatten())) + np.sum(np.square(R2.flatten())))/(2*m)
	
	#calculating true cost without regulation
	cost = np.sum(np.multiply(np.log(h2),Y)) + np.sum(np.multiply(np.log(1-h2),1-Y))
	cost = -cost/m

	#calculate total cost
	totalCost = cost + costRegulation

	return totalCost


def nnGradFunction(nnParams, X, y, lambdaVal):
	m = X.shape[0]
	theta1 = nnParams[0]
	n1 = X.shape[1] + 1
	n2 = theta1.shape[0]
	theta1 = theta1.reshape(int(n2/n1),n1)
	theta2 = nnParams[1]
	n1 = theta1.shape[0] + 1
	n2 = theta2.shape[0]
	theta2 = theta2.reshape(int(n2/n1),n1)
	

	#prepare Y matrix for cost function
	numLabels = np.unique(y).shape[0]+1
	#create boolean array of value or not out of 1s and 0s
	Y = (y==1).astype(int)
	for i in range(2,numLabels):
		Y = np.append(Y,(y==i).astype(int))
	#reshape so first dimension corresponds with label
	Y = Y.reshape(10,5000)

	X = np.insert(X,0,np.ones(X.shape[0]),axis=1) # adding bias unit
	h1 = helper.sigmoid(np.matmul(X,theta1.transpose()))
	h1 = np.insert(h1,0,np.ones(h1.shape[0]),axis=1) # adding bias unit
	h2 = helper.sigmoid(np.matmul(h1,theta2.transpose())).transpose()


	#calculate gradients
	theta2Error = h2-Y
	theta1Error = np.multiply(np.matmul(theta2Error.transpose(),theta2),np.multiply(h1,1-h1))
	theta1Grad = np.matmul(theta1Error.transpose(),X)
	theta1Grad = theta1Grad[1:,:]#drop bias unit error from hiddent layer
	theta2Grad = np.matmul(theta2Error,h1)
	

	return np.array([theta1Grad.flatten(), theta2Grad.flatten()])


def sigmoidGradient(Z):
	R = helper.sigmoid(Z)
	return np.multiply(R,1-R)



