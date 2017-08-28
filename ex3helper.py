import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import ex2helper as helper
import math
import matplotlib.image as mpimg

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
		print('cost for {} = {:.3f}, max = {:.3f}'.format(i%10,results.fun,np.max(p)))

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

def displayData(X, **keywordParameters):
	#set example width automatically if not given
	if('exampleWidth' in keywordParameters):
		exampleWidth = keywordParameters['exampleWidth']
	else:
		exampleWidth = round(math.sqrt(X.shape[1]))

	#calculate size of rows and columns
	[m, n] = X.shape
	exampleHeight = n//exampleWidth #eliminating float with // divide

	#calculate number of items to display
	displayRows = math.floor(math.sqrt(m))
	displayColumns = math.ceil(m/displayRows)

	#set padding between images
	padding = 1

	#set up blank display
	displayHeight = padding + displayRows * (exampleHeight + padding)
	displayWidth = padding + displayColumns * (exampleWidth + padding)

	displayArray = - np.ones([displayHeight, displayWidth])

	#Copy each example into a path on the display array
	currentExample = 0
	for j in range(0,displayRows):
		for i in range(0, displayColumns):
			if(currentExample > m):
				break

			#Copy the Patch

			#1. get the max value of the patch
			maxValue = np.amax(np.absolute(X[currentExample,:]))
			
			#2. get current example in the correct shape
			example = np.reshape(X[currentExample,:], [exampleHeight, exampleWidth])/maxValue
			example = example.transpose()

			#3. calculate current position height and width
			currentPositionHeight = padding + j * (exampleHeight + padding)
			currentPositionWidth = padding + i * (exampleWidth + padding)
			
			#4. assign current example to correct position in the display array
			displayArray[currentPositionHeight:currentPositionHeight + exampleHeight, currentPositionWidth:currentPositionWidth + exampleWidth] = example

			#5. iterate current example
			currentExample = currentExample + 1

		if(currentExample>m):
			break

	#show image
	imgplot = plt.imshow(displayArray, cmap='gray')
	plt.axis('off')
	plt.show()

def predict(theta1, theta2, X):
	m = X.shape[0]
	num_labels = theta2.shape[0]

	X = np.insert(X,0,np.ones(X.shape[0]),axis=1) # adding bias unit
	a1 = np.matmul(X,theta1.transpose())
	a1 = helper.sigmoid(a1)
	a1 = np.insert(a1,0,np.ones(a1.shape[0]),axis=1) # adding bias unit
	a2 = np.matmul(a1,theta2.transpose())
	a2 = helper.sigmoid(a2)
	
	return(np.argmax(a2,axis=1)+1)

