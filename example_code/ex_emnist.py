# example of loading the mnist dataset
from tensorflow.keras.datasets.mnist import load_data
from mnist import MNIST
from matplotlib import pyplot
import numpy
import random

mndata = MNIST('./emnist_data')
mndata.gz = True
mndata.select_emnist('byclass')

# load the images into memory
#(trainX, trainY) = mndata.load_training()
#trainX, trainY = numpy.asarray(trainX), numpy.asarray(trainY)
(testX, testY) = mndata.load_testing()
testX, testY = numpy.asarray(testX), numpy.asarray(testY)
testX = testX / 255.0
testX = numpy.reshape(testX, [-1, 28, 28])

#trainX, testX = trainX / 250.0, testX / 250.0
#(tX, tY), (teX, teY) = load_data()

# plot images from the training dataset
for i in range(25):
	# define subplot
	pyplot.subplot(5, 5, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	#img = numpy.array(numpy.zeros([28,28]))
	#for j,px in enumerate(testX[random.randint(0, len(testX))]):
	#	img[j//28][j%28] = px
	pyplot.imshow(testX[random.randint(0, len(testX))], cmap='gray_r')
pyplot.show()
