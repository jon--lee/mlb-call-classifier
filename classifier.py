# With 300 training examples: 200 epochs, 0.05 learning rate, 2-7-1 network
# for CB Bucknor works well


import numpy as np
import neuralpy
import matplotlib.pyplot as plt
import parser
import random
import json


# place that decimals should be rounded to (usually for output)
rounding_place = 0

# global network variable that is assigned in the train method
network = None


def test(net, data):
	"""
	@param net 		network to be tested
	@param data		list of tuples of input and output vectors.
					Output must be 1d vector.

	@return 		Returns the number of data where the expected output
					equals the output of the network given the corresponding input vector.

	
	"""
	count = 0
	for datum in data:
		inputs = datum[0]
		output = round(datum[1][0], rounding_place)
		actual = round(net.feedforward(inputs)[0], rounding_place)
		if(round(actual, rounding_place) == output):
			count += 1

	return count



def train(net, uris, epochs, learning_rate, validation_percentage, save_file = False):
	"""
	@param net 						network to be trained
	@param uris						uris that the data is parsed from
	@param epochs					maximum number of iterations
	@param learning_rate			learing rate
	@param validation_percentage	percentage of training_set that should be used for validation instead
	@param save_file				optional filepath to save weights and biases

	@postcondition: 				trains network with accurate weights and biases
									with given arguments (network not returned, object is just modified)
	@return 						percentage of correct test/verification data (see if it trained correctly)
	"""

	pitches = parser.parse(uris)

	inputs = np.array([ [pitch[0], pitch[1]] for pitch in pitches ])
	outputs = np.array([ [pitch[2]] for pitch in pitches ])

	inputs -= np.mean(inputs, axis=0)							# zeroing data
	inputs /= np.std(inputs, axis=0)							# normalizing data



	training_set = [ (i,o) for i, o in zip(inputs, outputs) ]	# data structure for data according to neuralpy
	random.shuffle(training_set)								# randomize the training set so not training same things in same order
	neuralpy.output("len: " + str(len(training_set)))

	cutoff = int(validation_percentage * len(training_set))		# determine the cutoff index

	test_set = training_set[:cutoff]							# fraction of all data that is test set
	training_set = training_set[cutoff:]						# training set being cut down to other fraction


	net.train(training_set, epochs, learning_rate, monitor_cost=True)


	# count = test(net, training_set)										# getting number of correct examples in training set
	count = test(net, test_set)												# number of correct examples in test_set


	# if there is a save file specified, save the weights
	# and biases to the file in json format.
	if save_file:
		save(save_file, net)

	network = net 															# setting the global variable for reuse
	return float(count)/len(test_set)


def save(filepath, net):
	"""
	@param filepath		filepath of the out file
	@param net 			network to be saved

	@postcondition		network layers, biases, and weights written in json
						to the output file.
	"""

	data = {
		"weights": [w.tolist() for w in net.weights],
		"biases": [b.tolist() for b in net.biases],
		"layers": net.layers
	}
	with open(filepath, 'w') as outfile:
		json.dump(data, outfile)




def graph_strikezone(net):
	"""
	@param net 			network to be tested

	@postcondition 		graph the strikezone with the given network or global network
						creates data points all around the strikezone (to demonstrate balls as well)
						also normalizes and zeroes the data
	"""

	increment = .01
	xs = np.arange(-1.8, 1.8, increment)
	xs -= np.mean(xs)
	xs /= np.std(xs)

	zs = np.arange(0, 3.9, increment)
	zs -= np.mean(zs)
	zs /= np.std(zs)

	c = np.zeros((len(zs), len(xs))).tolist()					# here is the matrix of zeros
																# rows represent the z and columns
																# represent the x
																# in the matrix, (x,y) is given as
																# [y][x]

	# c = np.array([[ [i, j] for i in xs] for j in zs ])
	#c = np.array([[ [i, j] for j in zs] for i in xs ])


	for i, ival in enumerate(zs):
		for j, jval in enumerate(xs):
			c[i][j] = round( net.feedforward([jval,ival])[0], rounding_place )

	c = np.array(c)
	c.transpose()

	xs = np.arange(-1.8, 1.8, increment)
	zs = np.arange(0, 3.9, increment)

	cs = plt.pcolor(xs, zs, c)
	plt.axis([xs.min(), xs.max(), zs.min(), zs.max()])
	plt.xlabel("distance from middle of plate, catcher's perspective (ft)")
	plt.ylabel("height (ft)")
	cb = plt.colorbar(cs)
	
	plt.show()

