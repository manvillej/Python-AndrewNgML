import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import ex2helper as helper

def OneVsAll(X, y, numlabels, lambdaVal):
	m = X.shape[0] #number of examples
	n = X.shape[1] #number of data points
 
	X = np.insert(X,0,np.ones(X.shape[0]),axis=1) # adding bias unit
	theta = np.array([])#initialize theta


	for i in numlabels:
		yTemp = np.zeros(y.shape[0])
		yTemp[np.where(y==i)] = 1
		thetaTemp = np.zeros(n + 1)

		#run regularized optimization
		results = helper.optimizeReg(thetaTemp, X, yTemp, lambdaVal)
		thetaTemp = results.x

		#get prediction accuracy
		p = helper.predict(thetaTemp, X)
		predictions = np.zeros(p.shape)
		predictions[np.where(p==yTemp)] = 1
		p = helper.sigmoid(np.matmul(X,thetaTemp))

		#calculating cost and accuracy to validate that the function is working correctly
		print('Train Accuracy: {:.1f}%'.format(np.mean(predictions) * 100))
		print('cost for {} = {:.3f}, max = {:.3f}'.format(i,results.fun,np.max(p)))

		theta = np.append(theta, thetaTemp)#appending discovered theta to theta

	#struggled on this for awhile. Reshape works from left to right, top to bottom. 
	#so if your data needs to be in columns instead of rows. It messes it all up, but it still works 
	theta = np.reshape(theta, (numlabels.shape[0],n + 1))
	return theta.transpose()

def predictOneVsAll(allTheta, X):
	X = np.insert(X,0,np.ones(X.shape[0]),axis=1) # adding bias unit

	pred = helper.sigmoid(np.matmul(X,allTheta))#calculate predictions for all thetas
	
	#return vector of position of maximum for each row +1 to adjust for arrays initializing at 0
	return(np.argmax(pred,axis=1)+1)
