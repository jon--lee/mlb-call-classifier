"""
Training classifier for Bill Miller's strikezone based
on game day from 2014 and 2015.
"""

import classifier
import neuralpy
import grapher

net = neuralpy.Network(2, 7, 1)

uris = [
	"miller_xml/1.xml",
	"miller_xml/2.xml",
	"miller_xml/3.xml"
]

epochs = 200
learning_rate = 0.05

validation_percentage = .32

neuralpy.output(classifier.train(net, uris, epochs, learning_rate, validation_percentage, save_file='results/miller.txt'))
classifier.graph_strikezone(net)

#grapher.graph(filepath='results/miller-93.txt')