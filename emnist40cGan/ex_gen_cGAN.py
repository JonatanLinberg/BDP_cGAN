# example of loading the generator model and generating images
from numpy import asarray
from numpy import zeros
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from sys import argv

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=62):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# create and save a plot of generated images
def save_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()

# load model
if (len(argv) > 1):
	f_name = argv[1]
else:
	f_name = input('enter generator file name: ')
model = load_model(f_name)

while(True):
	# generate images
	latent_points, labels = generate_latent_points(100, 100)
	# specify labels
	labels = asarray([x for _ in range(10) for x in range(10)])
	# generate images
	char = input('enter char id: ')
	labels = zeros(100)
	for i in range(100):
		labels[i] = char

	out = model.predict([latent_points, labels])
	out = (out + 1) / 2.0
	save_plot(out, 10)