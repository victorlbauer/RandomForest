import numpy as np
import math
from ID3 import *

class RandomForest():
	def __init__(self, path):
		self.path = path
		self.num_att = self.getNumAtt()
		self.dataset = self.loadDataset(path)
		self.ID3 = ID3()
		
	def getNumAtt(self):
		arq = open(self.path, 'r')
		line = arq.readline().split(',')
		num_att = len(line)
		arq.close()
		return num_att
	
	#Aloca os dados em uma matriz
	def loadDataset(self, path):
		arq = open(path,  'r')
		dataset = np.empty((0, self.num_att ))
		
		i = 0
		for line in arq:
			example = line.split(',')
			dataset = np.vstack((dataset, example))
			i += 1
		arq.close()
		self.num_examples = i
		return dataset
	
	#(P%) para treino, (1 - P)% para teste
	def setTrainingAndTesting(self, P):
		self.training_set = self.dataset[:int(math.ceil(self.num_examples*P))]
		self.testing_set = self.dataset[int(math.ceil(self.num_examples*P)):]

	def getTrainingSet(self):
		return self.training_set
	
	def shuffle(self):
		np.random.shuffle(self.dataset)
		
	#TODO: terminar a random forest. O ID3 esta pronto, ele retorna (printa e classifica, se quiser)	um dataset qualquer
	#Contudo, com o RandomForest temos que repetir esse processo N vezes para o mesmo dataset
	
	def train(self):
		pass
		
	def predict(self):
		pass
		
if __name__ == "__main__":
	rf = RandomForest("balloons.data.txt")
	rf.shuffle()
	#rf.setTrainingAndTesting(0.8) #80% treino, 10% teste
	rf.setTrainingAndTesting(1.0) #100% do dataset para treino, 0% teste
	
	training_set = rf.getTrainingSet()
	print training_set
	print
	tree = rf.ID3.train(training_set)
	rf.ID3.printTree(tree, 0)
	
	example = np.array(["YELLOW", "LARGE", "STRETCH", "ADULT"])
	rf.ID3.predict(tree, example)

