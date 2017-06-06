import numpy as np
import math
from ID3 import *

class RandomForest():
	def __init__(self, path, p_training, p_examples, p_att, n_trees):
		self.path = path
		self.p_training = p_training
		self.p_examples = p_examples
		self.p_att = p_att
		self.n_trees = n_trees
		
		self.num_att = self.getNumAtt()
		self.dataset = self.loadDataset(path)
		self.ID3 = ID3()
		
		self.shuffle()
		self.setTrainingAndTesting()

		
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
	def setTrainingAndTesting(self):
		self.training_set = self.dataset[:int(math.ceil(self.num_examples*self.p_training))]
		self.testing_set = self.dataset[int(math.ceil(self.num_examples*self.p_training)):]

	def getTestSet(self):
		return self.testing_set
	
	def permute(self, dataset):
		return np.random.permutation(dataset)
			
	def shuffle(self):
		np.random.shuffle(self.dataset)
	
	def getBatch(self, dataset, P):
		return dataset[:int(math.ceil(len(dataset)*P))]
	
	def splitClass(self, dataset):
		return dataset[:-1], dataset[-1] 
			
	def train(self):
		trees_list = []
		for _ in xrange(0, self.n_trees):
			ts = self.permute(self.training_set)
			batch = self.getBatch(ts, self.p_examples)
		
			sample_att, classes = self.splitClass(batch.T)
			sample_att = self.permute(sample_att)
			subsample = self.getBatch(sample_att, self.p_att)
		
			for att in subsample:
				classes = np.vstack((classes, att))
			dataset = np.fliplr(classes.T)
			
			tree = rf.ID3.train(dataset)
			trees_list.append(tree)
		self.trees_list = trees_list				
	
	def vote(self, class_dic):
		total = 0
		for entry in class_dic.keys():
			if(class_dic[entry] > total):
				total, pred = class_dic[entry], entry
		return pred
					
	def predict(self, example):
		class_dic = {}
		for tree in self.trees_list:
			prediction = rf.ID3.predict(tree, example)
			if(prediction not in class_dic.keys()):
				class_dic[prediction] = 1
			else:
				class_dic[prediction] += 1
		pred = self.vote(class_dic)
		print pred
					
if __name__ == "__main__":
	rf = RandomForest("balloons.data.txt", 0.8, 0.5, 0.5, 21)
	rf.train()
	test_set = rf.getTestSet()
	for example in test_set:
		rf.predict(example)
