import numpy as np
import scipy.optimize as op

def linearRegressionCost(theta, X, y, lambdaVal):
	m = y.shape[0]

	pred = linearRegressionPredict(X, theta)

	j = np.sum(np.power(pred-y,2))
	
	reg = np.power(theta,2)
	reg[0] = 0
	reg = lambdaVal*np.sum(reg)

	return (j+reg)/(2*m)

def linearRegressionGradient(theta, X, y, lambdaVal):
	m = y.shape[0]

	prediction = linearRegressionPredict(X, theta)
	
	error = prediction - y
	
	grad = np.matmul(X,error)/m

	reg = lambdaVal/m*theta
	reg[0] = 0

	return (grad + reg)

def linearRegressionPredict(X, theta):
	return np.matmul(X.transpose(),theta)

def trainLinearRegressionModel(theta, X, y, lambdaVal):
	return op.minimize(fun=linearRegressionCost, x0=theta, args=(X, y, lambdaVal), method='CG', jac=linearRegressionGradient)

def learningCurve(X,y,Xval,yval,lambdaVal):
	m = X.shape[1]
	theta = np.array([1, 1])

	errorTrain = np.array([])
	errorValidation = np.array([])
	for i in range(m):
		results = trainLinearRegressionModel(theta, X[:,:i+1], y[:i+1], lambdaVal)
		theta = results.x
		errorTrain = np.append(errorTrain,linearRegressionCost(theta, X[:,:i+1], y[:i+1], lambdaVal))
		errorValidation = np.append(errorValidation,linearRegressionCost(theta, Xval, yval, lambdaVal))

	return [errorTrain, errorValidation]
