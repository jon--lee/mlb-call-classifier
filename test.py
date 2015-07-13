import classifier
import json
import numpy as np
import neuralpy


with open('results/bucknor-94.txt') as data_file:    
	data = json.load(data_file)



weights = data['weights']
weights = [ np.array(w) for w in weights ]

biases = data['biases']
biases = [ np.array(b) for b in biases ]

net = neuralpy.Network(2, 7, 1)
net.biases = biases
net.weights = weights

classifier.graph_strikezone(net)
