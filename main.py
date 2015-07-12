# With 300 training examples: 200 epochs, 0.05 learning rate, 2-7-1 network



import numpy as np
import neuralpy
import matplotlib.pyplot as plt
import parser
import random

net = neuralpy.Network(2,7, 1)

url_list = "files.txt"
urls = open(url_list)

pitches = parser.parse(urls)

inputs = np.array([ [pitch[0], pitch[1]] for pitch in pitches ])
outputs = np.array([ [pitch[2]] for pitch in pitches ])

inputs -= np.mean(inputs, axis=0)
inputs /= np.std(inputs, axis=0)


training_set = [ (i,o) for i, o in zip(inputs, outputs) ]
random.shuffle(training_set)
test_set = training_set[300:]
training_set = training_set[:300]


point = 0



epochs = 200
learning_rate = .05


for i in xrange(0, 1):
	net.randomize_parameters()

	net.train(training_set, epochs, learning_rate, monitor_cost=True)

	neuralpy.output("for regular training set")
	count = 0
	for test in training_set:
		inputs = test[0]
		output = round(test[1][0], point)
		actual = round(net.feedforward(inputs)[0], point)
		#neuralpy.output(str(actual) + " should be " + str(output))
		if(round(actual, point) == output):
			count += 1

	neuralpy.output(str(count) + " of " + str(len(training_set)))


	neuralpy.output("\nfor test set")
	count = 0
	for test in test_set:
		inputs = test[0]
		output = round(test[1][0], point)
		actual = round(net.feedforward(inputs)[0], point)
		#neuralpy.output(str(actual) + " should be " + str(output))
		if(round(actual, point) == output):
			count += 1

	neuralpy.output(str(count) + " of " + str(len(test_set)))
	neuralpy.output()
	neuralpy.output()

xs = np.arange(-1.8, 1.8, .01)
xs -= np.mean(xs)
xs /= np.std(xs)

zs = np.arange(0, 3.9, .01)
zs -= np.mean(zs)
zs /= np.std(zs)

print xs.shape
print zs.shape

c = np.array([[ [i, j] for i in xs] for j in zs ])


result = np.array([[ round(net.feedforward(j)[0], point) for j in i] for i in c])
		
cs = plt.pcolor(result)
cb = plt.colorbar(cs)
plt.show()


