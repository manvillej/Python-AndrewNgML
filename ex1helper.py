import numpy as np
def computeCost(x,y,theta):
	
	j = 1/2*np.mean(np.power(np.matmul(x.transpose(),theta)-y,2))

	return j


def gradientDescent(x, y, theta, alpha, iterations):
	m = x.shape[1]
	costHistory = np.zeros(iterations)
	for i in range(iterations):

		error = np.matmul(x.transpose(),theta)-y
		gradient = np.dot(x,error)
		
		theta = theta - alpha*gradient/m
		costHistory[i]= computeCost(x,y,theta)



	return [theta, costHistory]

def normalize(x,mu,std):
	return (x - mu)/std


def featureNormalize(x):
	vfunc = np.vectorize(normalize)
	mu = np.mean(x,axis=0)
	sigma = np.std(x,axis=0)
	x_norm = vfunc(x,mu,sigma)

	return[x_norm, mu, sigma]

def computeCostMulti(x, y, theta):
	m=y.shape[0]
	j = np.sum(np.power(np.matmul(x,theta)-y,2))/(2*m)
	return j

def gradientDescentMulti(x, y, theta, alpha, num_iters):
	m = y.shape[0]
	j_history = np.zeros(num_iters)
	for i in range(0,num_iters):
		error = np.matmul(x,theta)-y
		theta = theta - alpha*np.dot(error,x)/m
		j_history[i] = computeCostMulti(x,y,theta)

	return [theta, j_history]
