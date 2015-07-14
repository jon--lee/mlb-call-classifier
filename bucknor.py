"""
Training classifier for CB Bucknor's strikezone based
on game day from 2014 and 2015.
"""
import classifier
import neuralpy
import grapher



# net = neuralpy.Network(2, 7, 1)

# uris = [ "bucknor_xml/" + str(i) + ".xml" for i in range(1,13) ]

# epochs = 200
# learning_rate = 0.05

# validation_percentage = .32			# percentage of training set that should be
# 									# be used as validation

# classifier.stand = "L"
# for i in range(0, 10):
# 	net.randomize_parameters()
# 	neuralpy.output(classifier.train(net, uris, epochs, learning_rate, validation_percentage, save_file='results/bucknor_' + str(i) + '.txt'))
# classifier.graph_strikezone(net)



grapher.graph(filepath='results/bucknor_2.txt')