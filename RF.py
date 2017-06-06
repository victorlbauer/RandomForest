import numpy as np
import math
from ID3 import *

class RandomForest():
	def __init__(self, dataset, p_examples, p_att, n_trees):
		self.dataset = dataset
		self.p_examples = p_examples
		self.p_att = p_att
		self.n_trees = n_trees
		self.ID3 = ID3()
		
	def permute(self, dataset):
		return np.random.permutation(dataset)
			
	def getBatch(self, dataset, P):
		return dataset[:int(math.ceil(len(dataset)*P))]
	
	def splitClass(self, dataset):
		return dataset[:-1], dataset[-1] 
			
	def train(self):
		trees_list = []
		for _ in xrange(0, self.n_trees):
			ts = self.permute(self.dataset)
			batch = self.getBatch(ts, self.p_examples)
		
			sample_att, classes = self.splitClass(batch.T)
			sample_att = self.permute(sample_att)
			subsample = self.getBatch(sample_att, self.p_att)
		
			for att in subsample:
				classes = np.vstack((classes, att))
			dataset = np.fliplr(classes.T)
			
			tree = self.ID3.train(dataset)
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
			prediction = self.ID3.predict(tree, example)
			if(prediction not in class_dic.keys()):
				class_dic[prediction] = 1
			else:
				class_dic[prediction] += 1
		pred = self.vote(class_dic)
		print pred
