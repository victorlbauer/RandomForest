import numpy as np
from ID3 import *
from RF import *

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

def accuracy(path):
	rf_acc = 0.0
	decision_tree_acc = 0.0
	i = 1.0
	
	arq = open(path, 'r')
	for line in arq:
		preds = line.split(', ')
		preds[-1] = preds[-1].replace('\n', '')
		
		if(preds[0] == preds[-1]):
			rf_acc += 1.0
		if(preds[1] == preds[-1]):
			decision_tree_acc += 1.0
		i += 1.0
	print "RF acc: ", float(rf_acc/i)
	print "ID3 acc: ", float(decision_tree_acc/i)
	
if __name__ == "__main__":
	#Por enquanto ta essa gambiarra para rodar, depois posso modificar para ser automatico, mas temos pouco tempo
	#Mudar o nome do arquivo caso queira usar outro dataset. Ex: "balloons.data.txt" -> "cars.data.txt"
	dataset, num_examples = loadDataset("cars.data.txt")
	training_set, testing_set = setTrainingAndTesting(dataset, num_examples, 0.8)
	
	#RandomForest(dataset, [porcento do dataset treino para ser dividido], [porcento dos atributos a serem divididos], [numero de arvores])
	rf = RandomForest(training_set, 0.6, 0.6, 3)
	rf.train()
	
	decision_tree = ID3()
	tree = decision_tree.train(training_set)

	#Treinar primeiro, depois descomente o codigo abaixo para testar a accuracia dos algoritmos
	#Mudar o nome para a accuracia do dataset. Ex: "balloons_acc.txt" -> "cars_acc.txt"
#	accuracy('balloons_acc.txt')

''' #Descomente os (') para treinar, verifique se o nome dos arquivos esta de acordo com os anteriores
	
	test_len = len(testing_set)
	i = 1
	arq = open("balloons_acc.txt", "w") #Mudar o nome para o arquivo certo
	for example in testing_set:
		print example
		print "itt: ", i, "/", test_len
		true_value = example[-1].replace('\n', '')
		rf_p = rf.predict(example).replace('\n', '')
		dt_p = decision_tree.predict(tree, example).replace('\n', '')
		
		print rf_p, dt_p, true_value
		
		arq.write("%s, " % rf_p)
		arq.write("%s, " % dt_p)
		arq.write("%s\n" % true_value)
		i += 1
	arq.close()
'''

