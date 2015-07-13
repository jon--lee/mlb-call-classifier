import classifier
import json
import numpy as np
import neuralpy



def graph(**kwargs):
	"""
	grapher.graph(filepath='network.txt') 	|	grapher.graph(net=network)

	takes either  "filepath" kwarg or "net" kwarg.
	if "filepath", provide a string path to the json file to be read.
	should contain 'weights', 'biases', 'layers'.

	if "net", provide a neuralpy network object

	uses classifier to graph the strike zone
	"""
	
	if 'filepath' in kwargs:

		filepath = kwargs['filepath']
		with open(filepath) as data_file:    
			data = json.load(data_file)
		
		weights = data['weights']
		weights = [ np.array(w) for w in weights ]
		biases = data['biases']
		biases = [ np.array(b) for b in biases ]
		layers = data['layers']

		net = neuralpy.Network(layers)

		net.biases = biases
		net.weights = weights	
	
	else:
		net = kwars['net']


	classifier.graph_strikezone(net)
