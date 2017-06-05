import numpy as np
import math

class ID3():
	def __init__(self):
		pass
	
	#Treina a arvore de decisao pelo ID3
	def train(self, dataset):
		tree = self.generateTree(dataset.T[:-1], dataset.T[-1])
		return tree
			
	#Calcula a entropia de um dataset
	def entropy(self, dataset):
		E = 0.0
		unique_inst, qnt_inst = np.unique(dataset, return_counts=True)
		freqs = (qnt_inst.astype(float)/len(dataset))
			
		for p in freqs:
			if(p != 0.0):
				E -= p*np.log2(p)
		return E
	
	#Calcula o ganho de um determinado atributo a partir do dataset
	def gain(self, dataset,  att):
		G = self.entropy(dataset)
		val, counts = np.unique(att, return_counts=True)
		freqs = (counts.astype(float)/len(att))
		
		for p, v in zip(freqs, val):
			G -= p*self.entropy(dataset[att == v])
		return G	
	
	#Cria um dicionario do dataset de forma que cada atributo vira uma chave contendo um array com as linhas em que esse atributo aparece no dataset
	#Ex: YELLOW = [0, 1, 2, 3] aparece nos exemplos 0, 1, 2 e 3
	def createDictionary(self, dataset):
		data_dic = {}
		i = 0
		for att in dataset:
			if (att not in data_dic.keys()):
				data_dic[att] = np.array([], dtype=np.int)
				data_dic[att] = np.append(data_dic[att], i)
			else:
				data_dic[att] = np.append(data_dic[att], i)
			i += 1
		return data_dic

	def isPure(self, dataset):
			if(len(set(dataset)) == 1):
				return True

	#Gera a arvore de decisao que sao diversos dicionarios encapsulados uns nos outros
	def generateTree(self, att_set, label_set):
		if(self.isPure(label_set) or len(label_set) == 0):
			return label_set
		
		gain = np.array([self.gain(label_set, att) for att in att_set])
		best_att_index = np.argmax(gain)
	
		sets = self.createDictionary(att_set[best_att_index])
		
		tree = {}
		for att, index in sets.items():
			label_subset = label_set.take(index, axis=0)
			att_subset = att_set.T.take(index, axis=0)
			tree[att, best_att_index] = self.generateTree(att_subset.T, label_subset)
		return tree

	def printTree(self, tree, itt):
		if(type(tree) is np.ndarray):
			print (itt*"\t") + tree[0]
		else:	
			for node in tree.keys():
				print (itt*"\t"), node
				next_itt = itt + 1
				self.printTree(tree[node], next_itt)

	def predict(self, tree, example):
		if(type(tree) is np.ndarray):
			print "Classe = ", tree[0]
		else:
			for att, index in tree.keys():
				if(att == example[index]):
					self.predict(tree[att, index], example)
			
