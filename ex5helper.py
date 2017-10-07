import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

def linearRegressionCost(theta, X, y, lambdaVal):
	'''
	Calculate the cost for the linear regression model
	'''
	m = y.shape[0]
	X = np.insert(X,0,np.ones(X.shape[1]),axis=0)

	pred = linearRegressionPredict(X, theta)

	j = np.sum(np.power(pred-y,2))
	
	reg = np.power(theta,2)
	reg[0] = 0
	reg = lambdaVal*np.sum(reg)

	return (j+reg)/(2*m)

def linearRegressionGradient(theta, X, y, lambdaVal):
	'''
	Calculate the gradient for the linear regression model
	'''
	m = y.shape[0]
	X = np.insert(X,0,np.ones(X.shape[1]),axis=0)
	
	prediction = linearRegressionPredict(X, theta)
	
	error = prediction - y
	
	grad = np.matmul(X,error)/m

	reg = lambdaVal/m*theta
	reg[0] = 0

	return (grad + reg)

def linearRegressionPredict(X, theta, **kwargs):
	'''
	predict the value for the provided linear regression model
	'''
	addBias = kwargs.pop('addBias', False)

	if(addBias):
		X = np.insert(X,0,np.ones(X.shape[1]),axis=0)

	return np.matmul(X.transpose(),theta)

def trainLinearRegressionModel(theta, X, y, lambdaVal):
	'''
	train the Linear Regression model
	'''
	return op.minimize(fun=linearRegressionCost, x0=theta, args=(X, y, lambdaVal), method='CG', jac=linearRegressionGradient)

def learningCurve(X,y,Xval,yval,lambdaVal):
	'''
	Iterate throught each number of possible learning sample sizes and
	calculate the error for the training and validation sets
	'''

	m = X.shape[1]
	theta = np.ones(X.shape[0]+1)

	errorTrain = np.array([])
	errorValidation = np.array([])
	for i in range(m):
		results = trainLinearRegressionModel(theta, X[:,:i+1], y[:i+1], lambdaVal)
		theta = results.x
		errorTrain = np.append(errorTrain,linearRegressionCost(theta, X[:,:i+1], y[:i+1], lambdaVal))
		errorValidation = np.append(errorValidation,linearRegressionCost(theta, Xval, yval, lambdaVal))

	return [errorTrain, errorValidation]

def polyFeatures(X,p):
	'''
	map the features of X to polynomial features for nonlinear solutions
	'''
	results = X

	for i in range(2,p+1):
		results = np.append(results,np.power(X,i), axis=0)

	return results

def featureNormalize(X,**kwargs):
	'''
	normalize X by subtracting the mean and dividing the results by the standard deviation
	'''

	mean = kwargs.pop('mean', X.mean(axis=1))
	sigma = kwargs.pop('sigma', np.std(X,axis=1))

	Xnormalized = (X.transpose()-mean)/sigma
	return Xnormalized.transpose()

def plotFit(minX, maxX, X, theta, p):
	'''
	plot the linear regression line values
	'''
	valueMap = np.arange(minX-15, maxX+25, .05)
	valueMap = valueMap[np.newaxis,:]
	
	valueMapPoly = polyFeatures(valueMap, p)

	#calculating mean and standard deviation for normalizing valueMap
	mean = Xpoly.mean(axis=1)
	sigma = np.std(Xpoly,axis=1)

	valueMapPoly = featureNormalize(valueMapPoly,mean=mean,sigma=sigma)

	projection = linearRegressionPredict(valueMapPoly,theta,addBias=True)

	plt.plot(valueMap[0,:],projection,  label = "Regression Line", color='red', linestyle='--')
	
def validationCurve(X,y,Xval,yval,lambdaVector):
	'''
	Iterate through lamdba values and calculate training error
	and validation error to choose appropriate value for lambda
	'''
	theta = np.ones(X.shape[0]+1)

	errorTrain = np.array([])
	errorValidation = np.array([])


	for lambdaVal in lambdaVector:

		results = trainLinearRegressionModel(theta, X, y, lambdaVal)
		theta = results.x
		errorTrain = np.append(errorTrain,linearRegressionCost(theta, X, y, lambdaVal))
		errorValidation = np.append(errorValidation,linearRegressionCost(theta, Xval, yval, lambdaVal))

	return [errorTrain, errorValidation]

	
