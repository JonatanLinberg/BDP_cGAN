# example of loading the generator model and generating images
print(' Character Generator',
	'\n[===================]\n')
from numpy import asarray
from numpy import array
from numpy import zeros
from numpy import transpose
from numpy import float32
from numpy import append
from numpy import full
from numpy import savetxt
from numpy import arange
from numpy.linalg import norm
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from sys import argv
from time import sleep

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
	result = transpose(result)
	if (input("save? y/n: ") == 'y'):
		savetxt('eucl_dist_%dsamples.txt' % ex_cl, result, delimiter=',')
	else:
		pyplot.figure(figsize=(12, 5))
		pyplot.boxplot(result)
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
	return -1

def placeTextInArray(textList, textWidth = -1):
	n_cols = 0
	for word in textList:
		if (len(word) > n_cols):
			n_cols = len(word)
	if (n_cols < textWidth):
		n_cols = textWidth

	i = 0
	change = False
	while (i != -1 and len(textList) > 1):
		if (len(textList[i]) + len(textList[i+1]) < n_cols):
			textList[i] = textList[i] + " " + textList[i+1]
			textList.pop(i+1)
			change = True
		i = (i+1)
		if (i >= len(textList)-1 and change):
			i = 0
			change = False
		elif (i >= len(textList)-1 and not change):
			i = -1
	space_map = zeros((len(textList), n_cols), dtype=bool)
	for i, _ in enumerate(space_map):
		for j, _ in enumerate(space_map[i]):
			try:
				if (textList[i][j] == ' '):
					space_map[i, j] = True
			except IndexError:
				space_map[i, j] = True
			except Exception as e:
				print(e)
				quit()
	return textList, space_map

class n_lat_pt_exception(Exception):
	def __init__(self, got, expected):
		self.got = got
		self.expected = expected

	def __str__(self):
		return 'Incorrect number of latent point dimensions!\nGot: ' + str(self.got) + '\nExpected: ' + str(self.expected)


def load_latent_point(latent_dim, n_samples, filename=""):
	in_lpt = zeros((n_samples, latent_dim), dtype='float32')
	if (filename != ""):
		with open(filename) as lpt_file:	
			i=0
			for line in lpt_file:
				if (line[0] == '#'):
					try:
						n_lptf = int(line[1:])
						if (n_lptf != latent_dim):
							raise n_lat_pt_exception(n_lptf, latent_dim)
					except Exception as ex:
						print(ex)
						quit()
				else:
					for pt in in_lpt:
						pt[i] = float(line)
					i += 1
	return in_lpt


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, lptf_base_fname=""):
	if (lptf_base_fname != ""):
		return load_latent_point(latent_dim, n_samples, lptf_base_fname)
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

def generate_latent_points_similar(latent_dim, n_samples, lptf_base_fname="", variation=0.4):
	pt = generate_latent_points(latent_dim, 1, lptf_base_fname)
	return (generate_latent_points(latent_dim, n_samples) * variation) + full((n_samples, latent_dim), pt[0])

def generate_latent_points_not_random(latent_dim, rows, cols, map_range, map_dim = [-1, -1], lptf_base_fname=""):
	lps = load_latent_point(latent_dim, rows*cols, lptf_base_fname)
	lps = lps.reshape(rows, cols, latent_dim)
	for i in range(rows):
		val_i = map_range*(i + 0.5 - rows*0.5) / rows
		for j in range(cols):
			val_j = map_range*(j + 0.5 - cols*0.5) / cols
			for k in range(latent_dim):
				if (map_dim[0] == -1 and map_dim[1] == -1):
					lps[i, j, k] = lps[i, j, k] + ((k%2 or rows<=1) * val_i + ((k+1)%2 or cols<=1) * val_j)# + (lps[i,j,k] / 10) * (val_i+3)/(i + 0.5)
				elif (k == map_dim[0]): # dim 0 means val_i for dim k==dim[0]
					lps[i, j, k] = lps[i, j, k] + (val_i)# + (lps[i,j,k] / 10) * (val_i+3)/(i + 0.5)
				elif (k == map_dim[1]): # dim 1 means val_j for dim k==dim[1]
					lps[i, j, k] = lps[i, j, k] + (val_j)# + (lps[i,j,k] / 10) * (val_i+3)/(i + 0.5)
				else:
					lps[i, j, k] = lps[i, j, k] + 0#(lps[i,j,k] / 10) * (val_i+3)/(i + 0.5)
	return lps.reshape(cols*rows, latent_dim)

def ascii_print(out, rows, cols):
	for i in range(rows):
		chars = out[i*cols:(i+1)*cols]
		for y in range(28):
			for c in chars:
				for x in range(28):
					px = c[y, x]
					if (px > 0.9):
						print(chr(0x2593), end='')
					elif (px > 0.5):
						print(chr(0x2592), end='')
					elif (px > 0.1):
						print(chr(0x2591), end='')
					else:
						print(' ', end='')
			print('')
		sleep(0.3)

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
in_text = False
lat_map_range = 0
in_char_id = None
text_width = -1
in_dim = [-1, -1]
latent_dim = 100
ascii_out = False
lptf_name = ""
text_var = 0.4

# load model
if (len(argv) > 1):
	for i, arg in enumerate(argv):
		if (arg[0] == '-'):
			for j, opt in enumerate(arg):
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
				elif (opt == 'x'):
					ascii_out = True
				elif (opt == 't'):
					in_text = True
				elif (opt == 'v'):
					try:
						text_var = float(argv[i+1])
					except:
						print('Invalid text variation level specification!\nusage:\n\t" -t <variation level> "')
				elif (opt == 'p'):
					try:
						lptf_name = argv[i+1]
					except:
						print('Could not load latent point from file!\nusage:\n\t"python gen_cGAN.py -p <path to .lptf file> "')
				elif (opt == 'w'):
					try:
						text_width = int(argv[i+1])
					except:
						print('Invalid text width!\nusage:\n\t"python gen_cGAN.py -w <text width>"')
				elif (opt == 'L'):
					try:
						lat_map_range = float(argv[i+1])*2
					except:
						print('Invalid latent map range!\nusage:\n\t"python gen_cGAN.py -L <lat_map_range>"')
				elif (opt == 'd'):
					try:
						if (arg[j+1] == 'x'):
							in_dim[1] = int(argv[i+1])
						elif (arg[j+1] == 'y'):
							in_dim[0] = int(argv[i+1])
						else:
							raise Exception
					except:
						print('Invalid latent dim specification!\nusage:\n\t" -dx <latent space dim> " or " -dy <latent space dim> "')
				elif (opt == 'C'):
					try:
						in_char_id = int(argv[i+1])
					except:
						print('Invalid char ID!\nusage:\n\t"python gen_cGAN.py -C <char ID>"')
				elif (opt == 'H'):
					print(	'" -f <model.h5> ":\t- Load generator from file "model.h5"', 
							'" -c <n_classes> ":\t- Integer, number of data classes',
							'" -C <n_classes> ":\t- Integer, character class',
							'" -r <n_rows> ":\t\t- Integer, number of rows to be generated',
							'" -t ":\t\t\t- Text generation mode',
							'" -v <variation level> ":\t- Float, variation level in text generation mode',
							'" -w <width>":\t\t- Integer, row width for text generation mode',
							'" -L <latent_map_range> ":\t- Float, latent space map range (-latent_map_range to +latent_map_range)',
							'" -dx <latent_dim> ":\t- Integer, specifies latent dimension for map dimension x',
							'" -dy <latent_dim> ":\t- Integer, specifies latent dimension for map dimension y',
							'" -e ":\t\t\t- Euclidean Box-Plot mode, calculates and shows euclidean distance in the generated images',
							'" -x ":\t\t\t- ASCII output mode',
							'" -p <.lptf file> ":\t- load a latent point from .lptf file',
							sep='\n')
					quit()
						
print('Use "python char_gen.py -H" for a list of all options\n')

if (f_name == ""):
	f_name = input('Enter generator file name: ')
if (n_classes == 0):
	n_classes=int(input('Enter number of classes: '))

model = load_model(f_name)
# This *should* work, if not -> comment line and use '-D'-option
latent_dim = model.layers[1].input_shape[0][1]

stop = False
while(not in_text):
	if (in_char_id == None):
		try:
			in_char = input('Enter char ID: ')
			char = int(in_char)
		except ValueError:
			char = to_label(in_char)
			print("Using char ID:", char, end='')
			if (char >= 0):
				print("(", label_arr[char], ")")
			else:
				print('')
		except Exception as e:
			print(e)
			quit()
	elif (not stop):
		char = in_char_id
		stop = True
	else:	# only do once when -C
		quit()
	# generate images
	latent_points = generate_latent_points(latent_dim, rows*n_classes, lptf_base_fname=lptf_name)
	if (lat_map_range != 0):
		if (char == -1):
			latent_points = latent_points.reshape((rows, n_classes, latent_dim))
			for c in range(n_classes):
				pts = generate_latent_points_not_random(latent_dim, rows, 1, lat_map_range, map_dim=in_dim, lptf_base_fname=lptf_name)
				for r in range(rows):
					latent_points[r, c] = pts[r]
			latent_points = latent_points.reshape((n_classes*rows, latent_dim))
		else:
			latent_points = generate_latent_points_not_random(latent_dim, rows, n_classes, lat_map_range, map_dim=in_dim, lptf_base_fname=lptf_name)
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
		if (not ascii_out):
			save_plot(out, rows, n_classes)
		else:
			ascii_print(out, rows, n_classes)

text = ['this', 'is', 'placeholder', 'text', 'this  text  is  unneccessarily  spaced           ']
# text string is not None
while (n_classes == 47):
	new_text = input("Enter text here: ").split()
	if (new_text != []):
		text = new_text

	print("Using text: ", text)
	text, space_arr = placeTextInArray(text, text_width)
	
	# text to int array
	labels = zeros((space_arr.shape), dtype=int)
	for i, _ in enumerate(labels):
		for j, _ in enumerate(labels[i]):	
			try:
				labels[i,j] = to_label(text[i][j])
			except:
				labels[i,j] = 0
			if (labels[i,j] == -1):
				labels[i,j] = 0
				space_arr[i,j] = True

	height, width = labels.shape[0], labels.shape[1]
	lat_pts = generate_latent_points_similar(latent_dim, height*width, lptf_name, variation=text_var)
	out = model.predict([lat_pts, labels.reshape(height*width)])
	out = (out +1) / 2.0
	for h in range(height):
		for w in range(width):
			if (space_arr[h, w]):
				out[h*width+w] = zeros((28, 28, 1), dtype=float32)
	if (not ascii_out):
		save_plot(out, height, width)
	else:
		ascii_print(out, height, width)

input('Sorry, "-t" is only available for 47-class models\n')