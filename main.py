"""
main.py

main method to run the neural network calculation
"""
import csv

import stock_loader
import network

n_input = 366
n_neutral_neuron = 100
n_output = 12
size = [n_input, n_neutral_neuron, n_output]
training_data, validation_data, test_data = stock_loader.load_data()
net = network.Network(size)

n_epoch = 300
n_batch_size = 100
coe_learn = 0.003
net.SGD(training_data, n_epoch, n_batch_size, coe_learn, test_data=test_data)

n_weights = len(size)-1
f = open('size.csv', 'ab')
dataWriter = csv.writer(f)
dataWriter.writerow(size)
f.close()
f = open('weights.csv', 'ab')
dataWriter = csv.writer(f)
for n in range(0, n_weights):
	for m in range(0, size[n+1]):
		outStr = []
		for l in range(0, size[n]):
			outStr.append(net.weights[n][m][l])
		dataWriter.writerow(outStr)
f.close()
f = open('biases.csv', 'ab')
dataWriter = csv.writer(f)
for n in range(0, n_weights):
	for m in range(0, size[n+1]):
		outStr = []
		outStr.append(net.biases[n][m][0])
		dataWriter.writerow(outStr)
f.close()
