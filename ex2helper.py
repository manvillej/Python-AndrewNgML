import numpy as np
import matplotlib.pyplot as plt

def plotData(X, Y):
	#np.extract(Y==1,X[0]) returns an array of values in X where the value is 1 in the same location in Y
	positiveExamples = plt.scatter(np.extract(Y==1,X[:,0]), np.extract(Y==1,X[:,1]), label = "Admitted", marker='o', color='b', s=10)
	negativeExamples = plt.scatter(np.extract(Y==0,X[:,0]), np.extract(Y==0,X[:,1]), label = "Not Admitted", marker='x', color='r', s=10)	
	plt.legend(handles=[positiveExamples, negativeExamples], loc='lower left')
	plt.xlabel('Exam 1 Score')
	plt.ylabel('Exam 2 Score')
	plt.title('Student Applicants\' Test Scores')
	plt.show()

def sigmoid(z):
	return np.array([1/(1+np.exp(-z))]).transpose()


def costFunction(theta, x, y):
	m = x.shape[0]
	z = sigmoid(np.matmul(x,theta))
	J = -1/m*(np.sum(np.log(z)*y+np.log(1-z)*(1-y)))
	grad = np.matmul(x.transpose(),z-y)/m

	return [J, grad]