"""
Training classifier for Bill Miller's strikezone based
on game day from 2014 and 2015.
"""

import classifier
import neuralpy
import grapher

net = neuralpy.Network(2, 8, 1)

uris = [ "miller_xml/" + str(i) + ".xml" for i in range(1,13) ]

epochs = 200
learning_rate = 0.05

validation_percentage = .32

ps = []

classifier.stand = "L"

for i in range(0, 10):
	net.randomize_parameters()
	p = classifier.train(net, uris, epochs, learning_rate, validation_percentage, save_file='results/miller_' + str(i) + '.txt')
	neuralpy.output(p)
	ps.append(p)


i = ps.index(max(ps))
neuralpy.output("\n\n" + str(max(ps)) + " at " + str(i))

grapher.graph(filepath='results/miller_' + str(i) + '.txt')


# grapher.graph(filepath='results/miller_4.txt')