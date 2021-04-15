# example of loading the mnist dataset
from tensorflow.keras.datasets.mnist import load_data
from mnist import MNIST
from matplotlib import pyplot
import numpy

mndata = MNIST('../emnist_data')
mndata.gz = True
mndata.select_emnist('byclass')

# load the images into memory
(testX, testY) = mndata.load_testing()
testX, testY = numpy.asarray(testX), numpy.asarray(testY)
testX = testX / 255.0
testX = numpy.reshape(testX, [-1, 28, 28])


# Get index of each class
example_indices = [None] * (max(testY) + 1)
for i in range(3500):
	example_indices[testY[i]] = i


# plot images from the training dataset
rows = 5
cols = 13
num_classes = len(example_indices)
for i in range(rows * cols):
	# define subplot
	pyplot.subplot(rows, cols, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(testX[example_indices[i % num_classes]], cmap='gray')
pyplot.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
pyplot.show()
