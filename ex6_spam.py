## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Imports
import ex6_spamHelper as spamHelper
import numpy as np
import scipy.io as io
from sklearn import svm
from data import vocab

## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.


print('Preprocessing sample email (emailSample1.txt)\n')

emailSample1 = open("./data/emailSample1.txt", 'r')

contents = emailSample1.read()
wordIndices = spamHelper.processEmail(contents)

print('Word Indices:')
for i in wordIndices:
	print('{:04d}'.format(i))

input('\nPart 1 completed. Program paused. Press enter to continue: ')

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n. 
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('\nExtracting features from sample email (emailSample1.txt)')

# Extract Features
featureVector = spamHelper.getEmailFeatures(wordIndices)

print('Feature Vector stats:')
print('Length: {}'.format(featureVector.size))
print('Sum: {:}'.format(int(np.sum(featureVector))))


input('\nPart 2 completed. Program paused. Press enter to continue: ')

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment

mat = io.loadmat('./data/spamTrain.mat')
X = mat['X']
y = mat['y'].astype(int).ravel()

print('\nTraining Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')

C = .1

model = svm.SVC(C=C, kernel='linear')
model.fit(X,y)

print('Training Accuracy: {:.2f}'.format(100*model.score(X,y)))

input('\nPart 3 completed. Program paused. Press enter to continue: ')

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment

mat = io.loadmat('./data/spamTest.mat')
Xtest = mat['Xtest']
ytest = mat['ytest'].astype(int).ravel()

print('\nTraining Accuracy: {:.2f}'.format(100*model.score(Xtest,ytest)))

input('\nPart 4 completed. Program paused. Press enter to continue: ')


## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtain the vocabulary list

coefficients = model.coef_.ravel()

sortedPosition = np.argsort(coefficients)
sortedPosition = np.flip(sortedPosition,0)


vocabList = vocab.dictionary
inverseDictionary = {v: k for k, v in vocabList.items()}
print('\nPosition, Coefficient, Word')
for i in sortedPosition[:15]:
	print('{:04d}, {:.3f}, {}'.format(i,coefficients[i],inverseDictionary[i]))


input('\nPart 5 completed. Program paused. Press enter to continue: ')


## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!

#read file
filename = "./data/emailSample1.txt"
email = open(filename, 'r')
contents = email.read()
email.close()


# Extract Features
wordIndices = spamHelper.processEmail(contents)
featureVector = spamHelper.getEmailFeatures(wordIndices)

# Expand features
featureVector = featureVector[np.newaxis,:]

prediction = model.predict(featureVector)

print('\nProcessed file: \"{}\"'.format(filename))
print('Spam Classification: {}'.format(prediction[0]))
print('(1 indicates spam, 0 indicates not spam)')


input('\nPart 6 completed. Program completed. Press enter to exit: ')