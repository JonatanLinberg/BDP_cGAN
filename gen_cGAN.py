# example of loading the generator model and generating images
from numpy import asarray
from numpy import zeros
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from sys import argv

n_classes=int(input('number of classes: '))

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# create and save a plot of generated images
def save_plot(examples, rows, cols):
	fig = pyplot.figure(figsize=(cols * (28/96), rows * (28/96)))
	# plot images
	for i in range(rows * cols):
		# define subplot
		pyplot.subplot(rows, cols, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')

	pyplot.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
	pyplot.show()

# load model
if (len(argv) > 1):
	f_name = argv[1]
else:
	f_name = input('enter generator file name: ')
model = load_model(f_name)

while(True):
	ex_per_class = 10
	# generate images
	latent_points, labels = generate_latent_points(100, ex_per_class*n_classes)
	# specify labels
	labels = zeros(n_classes*ex_per_class)
	# generate images
	char = int(input('enter char id: '))
#	test = asarray([17, 40, 45, 19, 50, 49, 36, 55, 36, 49])
	labels = zeros(ex_per_class*n_classes)
	for i in range(ex_per_class*n_classes):
		if (char >= 0 and char < n_classes):
			labels[i] = char
		elif (char == -1):
			labels[i] = i % n_classes
		elif (char >= -8):
			for j in range(7):
				if ((-2)-j == char):
					labels[i] = ((i + j*100) // 10) % n_classes
#	if (char == 980311):
#		for i in range(10):
#			labels[i*10:(i+1)*10] = test
	
	

	out = model.predict([latent_points, labels])
	out = (out + 1) / 2.0
	save_plot(out, ex_per_class, n_classes)