'''
creatVocabList is used to translate vocabList.txt into a python dictionary object. 
It will associate each word with a representational number in a python dictionary 
and use the PrettyPrint module to create a python file from the resulting dictionary
to be used in other files. 
'''
import os
import pprint

def main():
	pyDict = pyDictCreator('./vocab.txt')
	pyDict.creatDict()
	pyDict.createPyDict()



class pyDictCreator():
	def __init__(self,filePath):

		#if the file path is absolute, make it an absolute path
		if(not os.path.isabs(filePath)):
			filepath = os.path.abspath(filePath)


		#set variables
		self.directory = os.path.dirname(filepath)
		self.basename = os.path.basename(filepath)
		self.filepath = filepath

		#sets up empty dictionary
		self.dictionary = {}
		self.inverseDictionary = {}

	def setFilePath(self, filepath):
		#if the file path is absolute, make it an absolute path
		if(not os.path.isabs(filePath)):
			filepath = os.path.abspath(filePath)


		#set variables
		self.directory = os.path.dirname(filepath)
		self.basename = os.path.basename(filepath)
		self.filepath = filepath

		#sets up empty dictionary
		self.dictionary = {}

	def creatDict(self):
		dictionary = {}

		txtFile = open(self.filepath, 'r')

		#get the dictionary words
		words = txtFile.readlines()
		txtFile.close()

		#value to start representing numbers
		#iterate through words adding them to the dictionary
		for word in words:
			value = word.split()
			dictionary[value[1]] = int(value[0])

		self.dictionary = dictionary

		return dictionary


	def createPyDict(self):
		os.chdir(self.directory)

		#open  new file with the same name as the original, but with extension .py
		name = self.basename.split('.')
		name = name[0]  + '.py'
		pyDict = open(name, 'w')


		pyDict.write('dictionary = ' + pprint.pformat(self.dictionary) + '\n')
		pyDict.write('dictionary = ' + pprint.pformat(self.dictionary) + '\n')
		pyDict.close()



if __name__ == '__main__':
	main()