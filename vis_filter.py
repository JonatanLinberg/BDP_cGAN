# cannot easily visualize filters lower down
from tensorflow.keras.models import load_model
from keras.models import Model
from matplotlib import pyplot
from numpy.random import randn
from numpy import array
from numpy import zeros
from numpy import reshape
from math import ceil
from sys import argv

MODEL_PATH = "final_experiment/exmo11/a/best_43.h5"
N_LATENT_DIM = 100
CLASS_ID = 10


def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

try:
	layer = int(argv[1])
except:
	layer = -1

# load the model
model = load_model(MODEL_PATH)
model = Model(inputs=model.inputs, outputs=model.layers[layer].output)

latent_pts = generate_latent_points(N_LATENT_DIM, 1) 
#latent_pts = zeros([1, N_LATENT_DIM], dtype="float32")
out = model.predict([latent_pts, array([CLASS_ID])])
print(model.layers[-1])
print(out.shape)
rows = 8

try:
	for i in range(out.shape[3]):
		ax = pyplot.subplot(rows, ceil(out.shape[3]/rows), i+1)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(out[0, :, :, i], cmap="gray")
	pyplot.show()
except:
	print(out)
	#out = reshape(out, [1, 1, 7, 7])
	pyplot.imshow(out[0], cmap="gray")
	pyplot.show()