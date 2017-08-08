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


	