# example of loading the generator model and generating images
from numpy import asarray
from numpy import array
from numpy import zeros
from numpy import transpose
from numpy import float32
from numpy import append
from numpy.linalg import norm
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from sys import argv

def plot_euclidean_distance(examples, n_cl):
	n_ex = len(examples)
	ex_cl = n_ex // n_cl
	ed_cl = ex_cl*(ex_cl-1)//2
	result = array(zeros((n_cl, ed_cl)), float32)
	# all same class
	for i in range(n_ex):
		for j in range((i//n_cl) + 1, n_ex//n_cl):
			i_cl = i // n_cl
			index = (j-i_cl -1) + (ed_cl - ((ex_cl-i_cl)*(ex_cl-i_cl-1)//2))
			#print('x:', i%n_cl, '\ty:', index, '\tj: ', j, '\tx: ', ex_cl, '\ted: ', ed_cl, '\ti_cl: ', i_cl, '\th(x-i_cl): ', ((ex_cl-i_cl)*(ex_cl-i_cl-1)//2))
			result[i % n_cl, index] = (norm(examples[i, :, :, 0] - examples[((j*n_cl)+(i%n_cl)), :, :, 0]))
	pyplot.figure(figsize=(12, 5))
	pyplot.boxplot(transpose(result))
	pyplot.xlabel("class")
	pyplot.ylabel("euclidean distance")
	pyplot.tight_layout()
	pyplot.ylim(bottom=0)
	pyplot.show()


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
	pyplot.figure(figsize=(cols * (28/96), rows * (28/96)))
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

eucl = False
f_name = ""
n_classes = 0
rows = 10

# load model
if (len(argv) > 1):
	for i, arg in enumerate(argv):
		if (arg[0] == '-'):
			for opt in arg:
				if (opt == 'r'):
					try:
						rows = int(argv[i+1])
					except:
						print('Invalid number of rows!\nusage:\n\t"py gen_cGAN.py -r <number of rows>"')
				elif (opt == 'e'):
					eucl = True
				elif (opt == 'c'):
					try:
						n_classes = int(argv[i+1])
					except:
						print('Invalid number of classes!\nusage:\n\t"py gen_cGAN.py -c <n_classes>"')
						n_classes = 0
				elif (opt == 'f'):
					try:
						f_name = argv[i+1]
						open(f_name)
					except:
						print('Invalid file name!\nusage:\n\t"py gen_cGAN.py -f <f_name>"')
						f_name = ""
						

if (f_name == ""):
	f_name = input('enter generator file name: ')
if (n_classes == 0):
	n_classes=int(input('number of classes: '))

model = load_model(f_name)

while(True):
	# generate images
	latent_points, labels = generate_latent_points(100, rows*n_classes)
	# specify labels
	labels = zeros(n_classes*rows)
	# generate images
	char = int(input('enter char id: '))
#	test = asarray([17, 40, 45, 19, 50, 49, 36, 55, 36, 49])
	labels = zeros(rows*n_classes)
	for i in range(rows*n_classes):
		if (char >= 0 and char < n_classes):
			labels[i] = char
			e_classes = 1
		elif (char == -1):
			labels[i] = i % n_classes
			e_classes = n_classes
		elif (char >= -8):
			for j in range(7):
				if ((-2)-j == char):
					labels[i] = ((i + j*100) // 10) % n_classes
#	if (char == 980311):
#		for i in range(10):
#			labels[i*10:(i+1)*10] = test
	out = model.predict([latent_points, labels])
	out = (out + 1) / 2.0

	if (eucl):
		plot_euclidean_distance(out, e_classes)
	else:
		save_plot(out, rows, n_classes)