# Training classifier for CB Bucknor's strikezone based
# on game day from 2014 and 2015.
import classifier
import neuralpy

net = neuralpy.Network(2, 7, 1)
uris = [
	"bucknor_xml/1.xml",
	"bucknor_xml/2.xml",
	"bucknor_xml/3.xml"
]

epochs = 200
learning_rate = 0.05

validation_percentage = .32			# percentage of training set that should be
									# be used as validation


neuralpy.output(classifier.train(net, uris, epochs, learning_rate, validation_percentage))
classifier.graph_strikezone(net)