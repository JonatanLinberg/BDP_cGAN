# example of loading the generator model and generating images
from numpy import asarray
from numpy import array
from numpy import zeros
from numpy import transpose
from numpy import float32
from numpy import append
from numpy import full
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


label_arr = (	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
				'A','B',('C','c'),'D','E','F','G','H',('I','i'),('J','j'),('K','k'),('L', 'l'),('M', 'm'),'N',('O', 'o'),('P','p'),'Q','R',('S', 's'),'T',('U', 'u'),('V', 'v'),('W', 'w'),('X', 'x'),('Y', 'y'),('Z', 'z'),
				'a','b','d','e','f','g','h','n','q','r','t')
def to_label(in_lbl):
	for i,a in enumerate(label_arr):
		if (type(a) is tuple):
			for b in a:
				if (in_lbl == b):
					return i
		elif (in_lbl == a):
			return i
	return 0

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

def generate_latent_points_similar(latent_dim, n_samples):
	pt = generate_latent_points(latent_dim, 1)
	return (generate_latent_points(latent_dim, n_samples) / 2.5) + full((n_samples, latent_dim), pt[0])

def generate_latent_points_not_random(latent_dim, rows, cols, map_range):
	lps = generate_latent_points(latent_dim, rows*cols)
	lps = lps.reshape(rows, cols, latent_dim)
	for i in range(rows):
		val_i = map_range*(i + 0.5 - rows*0.5) / rows
		for j in range(cols):
			val_j = map_range*(j + 0.5 - cols*0.5) / cols
			for k in range(latent_dim):
				lps[i, j, k] = ((k%2 or j<=1) * val_i + ((k+1)%2 or i<=1) * val_j) + (lps[i,j,k] / 10) * (val_i+3)/(i + 0.5)
	return lps.reshape(cols*rows, latent_dim)


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
in_text = None
lat_map_range = 0

# load model
if (len(argv) > 1):
	for i, arg in enumerate(argv):
		if (arg[0] == '-'):
			for opt in arg:
				if (opt == 'r'):
					try:
						rows = int(argv[i+1])
					except:
						print('Invalid number of rows!\nusage:\n\t"python gen_cGAN.py -r <number of rows>"')
				elif (opt == 'e'):
					eucl = True
				elif (opt == 'c'):
					try:
						n_classes = int(argv[i+1])
					except:
						print('Invalid number of classes!\nusage:\n\t"python gen_cGAN.py -c <n_classes>"')
						n_classes = 0
				elif (opt == 'f'):
					try:
						f_name = argv[i+1]
						open(f_name)
					except:
						print('Invalid file name!\nusage:\n\t"python gen_cGAN.py -f <f_name>"')
						f_name = ""
				elif (opt == 't'):
					try:
						in_text = argv[i+1].split()
					except:
						in_text = []
				elif (opt == 'L'):
					try:
						lat_map_range = int(argv[i+1])
					except:
						print('Invalid latent map range!\nusage:\n\t"python gen_cGAN.py -L <lat_map_range>"')

						

if (f_name == ""):
	f_name = input('Enter generator file name: ')
if (n_classes == 0):
	n_classes=int(input('Enter number of classes: '))

model = load_model(f_name)

while(in_text == None):
	try:
		in_char = input('Enter char ID: ')
		char = int(in_char)
	except ValueError:
		char = to_label(in_char)
		print("Using char ID:", char, "(", label_arr[char], ")")
	except Exception as e:
		print(e)
		quit()

	# generate images
	latent_points = generate_latent_points(100, rows*n_classes)
	if (lat_map_range != 0):
		if (char == -1):
			latent_points = latent_points.reshape((rows, n_classes, 100))
			for c in range(n_classes):
				pts = generate_latent_points_not_random(100, rows, 1, lat_map_range)
				for r in range(rows):
					latent_points[r, c] = pts[r]
			latent_points = latent_points.reshape((n_classes*rows, 100))
		else:
			latent_points = generate_latent_points_not_random(100, rows, n_classes, lat_map_range)
	# specify labels
	labels = zeros(n_classes*rows)
	# generate images
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

text = ['this', 'is', 'placeholder', 'text']
# text string is not None
while (n_classes == 47):
	if (in_text == []):
		new_text = input("Enter text here: ").split()
		if (new_text != []):
			text = new_text
	else:
		text = in_text
	n_cols = 0
	for word in text:
		if (len(word) > n_cols):
			n_cols = len(word)

	# text to int array
	labels = zeros((len(text), n_cols), dtype=int)
	for i, _ in enumerate(labels):
		for j, _ in enumerate(labels[i]):
			try:
				labels[i,j] = to_label(text[i][j])
			except:
				labels[i,j] = 0
	height, width = labels.shape[0], labels.shape[1]
	lat_pts = generate_latent_points_similar(100, height*width)
	out = model.predict([lat_pts, labels.reshape(height*width)])
	out = (out +1) / 2.0
	for h in range(height):
		for w in range(width):
			if (len(text[h]) <= w):
				out[h*width+w] = zeros((28, 28, 1), dtype=float32)
	save_plot(out, height, width)

input('Sorry, "-t" is only available for 47-class models\n')