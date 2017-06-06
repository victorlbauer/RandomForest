import numpy as np
from ID3 import *
from RF import *

#pegar o path para rodar
#dividir o dataset
#passar o dataset de treino para o ID3 e RF

def getNumAtt(path):
	arq = open(path, 'r')
	line = arq.readline().split(',')
	num_att = len(line)
	arq.close()
	return num_att

def loadDataset(path):
	num_att = getNumAtt(path)
	arq = open(path,  'r')
	dataset = np.empty((0, num_att))		
	
	i = 0
	for line in arq:
		example = line.split(',')
		dataset = np.vstack((dataset, example))
		i += 1
	arq.close()
	num_examples = i
	return dataset, num_examples

def permute(dataset):
	return np.random.permutation(dataset)

def setTrainingAndTesting(dataset, num_examples, P):
	shuffled_data = permute(dataset)
	training_set = shuffled_data[:int(math.ceil(num_examples*P))]
	testing_set = shuffled_data[int(math.ceil(num_examples*P)):]
	return training_set, testing_set
	
if __name__ == "__main__":
	dataset, num_examples = loadDataset("balloons.data.txt")
	training_set, testing_set = setTrainingAndTesting(dataset, num_examples, 0.8)

	rf = RandomForest(training_set, 0.5, 0.5, 21)
	rf.train()
	
	decision_tree = ID3()
	tree = decision_tree.train(training_set)
	
	
