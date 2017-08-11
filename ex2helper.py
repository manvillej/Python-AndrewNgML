import numpy as np
import matplotlib.pyplot as plt

def plotData(X, Y):
	positiveExamples = plt.scatter(np.extract(Y==1,X[0]), np.extract(Y==1,X[1]), label = "Positive Examples", marker='x', color='r', s=10)
	negativeExamples = plt.scatter(np.extract(Y==0,X[0]), np.extract(Y==0,X[1]), label = "Negative Examples", marker='o', color='b', s=10)	
	plt.legend(handles=[positiveExamples, negativeExamples], loc='lower left')
	plt.xlabel('X[0]')
	plt.ylabel('X[1]')
	plt.title('Raw Data')
	plt.show()