"""
main.py

main method to run the neural network calculation
"""
import stock_loader
import network


n_input = 366
n_neutral_neuron = 100
n_output = 12
n_epoch = 100
n_batch_size = 500
coe_learn = 0.01
size = [n_input, n_neutral_neuron, n_output]

for x in range(10):
	print "cycle number : {0:03d}".format(x+1)
	training_data, validation_data, test_data = stock_loader.load_data()
	net = network.Network(size)
	net.SGD(training_data, n_epoch, n_batch_size, coe_learn, test_data=test_data)

