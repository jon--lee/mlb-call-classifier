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

# side of the plate that the batter stands on (by default it is right)
stand = "R"

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
	global net
	global mean
	global std

	pitches = parser.parse(uris, stand)

	inputs = np.array([ [pitch[0], pitch[1]] for pitch in pitches ])
	outputs = np.array([ [pitch[2]] for pitch in pitches ])

	mean = np.mean(inputs, axis=0)
	std = np.std(inputs, axis=0)

	inputs -= mean												# zeroing data
	inputs /= std 												# normalizing data



	training_set = [ (i,o) for i, o in zip(inputs, outputs) ]	# data structure for data according to neuralpy
	random.shuffle(training_set)								# randomize the training set so not training same things in same order
	neuralpy.output("len: " + str(len(training_set)))

	cutoff = int(validation_percentage * len(training_set))		# determine the cutoff index

	test_set = training_set[:cutoff]							# fraction of all data that is test set
	training_set = training_set[cutoff:]						# training set being cut down to other fraction

	batch_length = int(.6 * len(training_set))

	net.train(training_set, epochs, learning_rate, batch_length=batch_length, monitor_cost=True)


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
		"layers": net.layers,
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
	x_edge = 1.8
	z_edge = 3.9

	xs = np.arange(-x_edge, x_edge, increment)
	xs -= np.mean(xs)
	xs /= np.std(xs)

	zs = np.arange(0, z_edge, increment)
	zs -= np.mean(zs)
	zs /= np.std(zs)


	c = np.zeros((len(xs), len(zs)))

	for i, ival in enumerate(xs):
		for j, jval in enumerate(zs):
			c[i][j] = round(net.feedforward([ival, jval])[0], rounding_place)


	# matplotlab plots matrices by a (y, x) as if rows
	# are the y and columns are the x, so we must transpose
	c = c.transpose()

	
	# zone = find_zone(increment, x_edge, len(xs), z_edge, len(zs))


	xs = np.arange(-x_edge, x_edge, increment)
	zs = np.arange(0, z_edge, increment)

	cs = plt.pcolor(xs, zs, c)
	plt.axis([xs.min(), xs.max(), zs.min(), zs.max()])
	plt.xlabel("distance from middle of plate, catcher's perspective (ft)")
	plt.ylabel("height (ft)")
	cb = plt.colorbar(cs)
	
	plt.show()


# def find_zone(increment, x_edge, x_len, z_edge, z_len):
# 	"""
# 	Determine the indices of the edges of the strike zone
# 	average height of top of strikezone is 3.5 ft
# 	average height of bottom of strikezone is 1.5ft

# 	@param increment		increment used for range
# 	@param x_edge			in feet, the distance from the middle of
# 								home plate to the edge of the view (assuming middle 
# 								of plate is middle of view as well)
# 	@param x_len			total length of the array of x values
# 	@param z_edge			in feet, the top of the view assuming 0 is the bottom
# 	@param z_len			total length of the z values
# 	"""
# 	z_scale = (z_len - 1) / z_edge			# (array len / feet)

# 	zone_top = z_scale * 3.5				# (array len / feet) * feet
# 	zone_bottom = z_scale * 1.5				# (array len / feet) * feet

# 	print zone_top
# 	print zone_bottom
