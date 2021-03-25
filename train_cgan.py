import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy import array
from numpy import reshape
from numpy import transpose
from numpy import float32
from numpy.random import randn
from numpy.random import randint
from numpy.linalg import norm
import os
# Remove # of the following two lines to prohibit usage of a GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from matplotlib import pyplot
from shutil import copyfile
from shutil import move
from ast import literal_eval
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from scipy.linalg import sqrtm
from skimage.transform import resize
from mnist import MNIST
from sys import argv

mndata = MNIST('./emnist_data')
mndata.gz = True

# SETUP
################################
# rtp_ = runtime parameter
# Folder and filename
rtp_folder_name = input("enter name: ") + '/'
rtp_root_folder = rtp_folder_name
# Dataset
# Note! Training or testing is set in the load_real_samples function
mndata.select_emnist('balanced')	# 'balanced', 'byclass'...
rtp_n_classes = 47		# Important, will crash if not set correctly
# size of the latent space
n_latent_dim = 100

rtp_def_conf = {'d_embedding':50,
				'd_hidden_layers1':0,
				'd_hidden_units1':0,
				'd_LeReLU_alpha':0.2,
				'd_conv_filters':64,
				'g_embedding':50,
				'g_hidden_layers1':1,
				'g_hidden_layers2':1,
				'g_hidden_units_mult2':128,
				'g_deconv_filters':128,
				'g_LeReLU_alpha':0.2,
				'SGD':'n',
				'SGD_momentum':0.0,
				'SGD_nesterov':'n',
				'learn_rate':0.0002}
rtp_conf_list = []
rtp_list_index = 0
for argi in range(1, len(argv)): # linked rtp-file
	print('Reading RTPs from file: ', argv[argi])
	with open(argv[argi], 'r') as rtp_file:
		# Discriminator parameters
		rtp_conf = {}
		rtp_conf['d_embedding'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf['d_hidden_layers1'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf['d_hidden_units1'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf['d_LeReLU_alpha'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf['d_conv_filters'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		# Generator parameters
		rtp_conf['g_embedding'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf['g_hidden_layers1'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf['g_hidden_layers2'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf['g_hidden_units_mult2'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf['g_deconv_filters'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf['g_LeReLU_alpha'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		# other
		rtp_conf['SGD'] = rtp_file.readline().split(':')[1].strip()
		if (rtp_conf['SGD'] == 'y'):
			rtp_conf['SGD_momentum'] = literal_eval(rtp_file.readline().split(':')[1].strip())
			rtp_conf['SGD_nesterov'] = rtp_file.readline().split(':')[1].strip()
		rtp_conf['learn_rate'] = literal_eval(rtp_file.readline().split(':')[1].strip())
		rtp_conf_list.append(rtp_conf)

if (len(rtp_conf_list) == 0):
	try:# Discriminator parameters
		rtp_conf = {}
		print(" < < Discriminator > >")
		rtp_conf['d_embedding'] = literal_eval(input('<D> embedding parameters (50): ').strip())
		rtp_conf['d_hidden_layers1'] = literal_eval(input('<D> hidden layers1 (0): ').strip())
		rtp_conf['d_hidden_units1'] = literal_eval(input('<D> hidden units1 (0): ').strip())
		rtp_conf['d_LeReLU_alpha'] = literal_eval(input('<D> leaky ReLU alpha (0.2): ').strip())
		rtp_conf['d_conv_filters'] = literal_eval(input('<D> convolution filters (128): ').strip())
		# Generator parameters
		print(" < < Generator > >")
		rtp_conf['g_embedding'] = literal_eval(input('<G> embedding parameters (50): ').strip())
		rtp_conf['g_hidden_layers1'] = literal_eval(input('<G> hidden layers1 (1): ').strip())
		rtp_conf['g_hidden_layers2'] = literal_eval(input('<G> hidden layers2 (1): ').strip())
		rtp_conf['g_hidden_units_mult2'] = literal_eval(input('<G> hidden unit mult2 (128): ').strip())
		rtp_conf['g_deconv_filters'] = literal_eval(input('<G> deconvolution filters (128): ').strip())
		rtp_conf['g_LeReLU_alpha'] = literal_eval(input('<G> leaky ReLU alpha (0.2): ').strip())

		print(" < < Other > >")
		rtp_conf['SGD'] = input('SGD (y/n): ').strip()
		if (rtp_conf['SGD'] == 'y'):
			rtp_conf['SGD_momentum'] = literal_eval(input('SGD momentum (0.0): ').strip())
			rtp_conf['SGD_nesterov'] = input('SGD nesterov (y/n): ').strip()
		rtp_conf['learn_rate'] = float(input('Learning rate (0.0002): ').strip())
		rtp_conf_list.append(rtp_conf)
	except: 
		print("ERROR!\nUsing default values...")
		rtp_conf_list.append(rtp_def_conf)


visualize = (input('Only show models (y/n): ') == 'y')

rtp_mode_collapse_lim = 2.0

# Training parameters
rtp_fid_samples = 25		# number of fid-batch-samples
rtp_train_n_batch = 128		# multiple of 16
# > > > rtp_train_n_batch * rtp_fid_samples ≈ 5120
rtp_train_n_epochs = 100

def untuple_list(l1):
	l2 = []
	for conf in l1:
		tupless = True
		for i in conf:
			if (type(conf[i]) is tuple):
				tupless = False
				for val in conf[i]:
					temp = conf.copy()
					temp[i] = val
					l1.append(temp)
		if (tupless and conf not in l2):
			l2.append(conf)

	return l2

rtp_conf_list = untuple_list(rtp_conf_list)

# Create directory for next run
os.makedirs(rtp_folder_name)
copyfile(argv[0].split('/')[len(argv[0].split('/'))-1], rtp_folder_name + argv[0].split('/')[len(argv[0].split('/'))-1])

# write RTPs to rtp.txt
for i, conf in enumerate(rtp_conf_list):
	with open(rtp_folder_name + 'rtp%d.txt' % i, 'w') as rtp_f:
		rtp_f.write('<D> embedding parameters (50):%d\n' % conf['d_embedding'])
		rtp_f.write('<D> hidden layers1 (0):%d\n' % conf['d_hidden_layers1'])
		rtp_f.write('<D> hidden units1 (0):%d\n' % conf['d_hidden_units1'])
		rtp_f.write('<D> leaky ReLU alpha (0.2):%f\n' % conf['d_LeReLU_alpha'])
		rtp_f.write('<D> convolution filters (128):%d\n' % conf['d_conv_filters'])
		rtp_f.write('<G> embedding parameters (50):%d\n' % conf['g_embedding'])
		rtp_f.write('<G> hidden layers1 (1):%d\n' % conf['g_hidden_layers1'])
		rtp_f.write('<G> hidden layers2 (1):%d\n' % conf['g_hidden_layers2'])
		rtp_f.write('<G> hidden units2 (128):%d\n' % conf['g_hidden_units_mult2'])
		rtp_f.write('<G> deconvolution filters (128):%d\n' % conf['g_deconv_filters'])
		rtp_f.write('<G> leaky ReLU alpha (0.2):%f\n' % conf['g_LeReLU_alpha'])
		rtp_f.write('SGD (y/n):' + conf['SGD'] + '\n')
		if (conf['SGD'] == 'y'):
			rtp_f.write('SGD momentum (0.0):%f\n' % conf['SGD_momentum'])
			rtp_f.write('SGD nesterov (y/n):' + conf['SGD'] + '\n')
		rtp_f.write('Learning rate (0.0002):%f\n' % conf['learn_rate'])
		rtp_f.write('n_classes:%d\n' % rtp_n_classes)


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# number of handshakes between 'a' people
def n_hs(a):
	return a*(a-1)//2

def calc_euclidean_distance(examples, n_cl, epoch):
	n_ex = len(examples)	# total examples
	ex_cl = n_ex // n_cl	# examples per class
	ed_cl = n_hs(ex_cl)		# number of "handshakes" per class
	result = array(zeros((n_cl, ed_cl)), float32)
	# all same class
	for i in range(n_ex):
		for j in range((i//n_cl) + 1, n_ex//n_cl):
			i_cl = i // n_cl
			index = (j-i_cl -1) + (ed_cl - n_hs(ex_cl-i_cl))
			#print('x:', i%n_cl, '\ty:', index, '\tj: ', j, '\tx: ', ex_cl, '\ted: ', ed_cl, '\ti_cl: ', i_cl, '\th(x-i_cl): ', ((ex_cl-i_cl)*(ex_cl-i_cl-1)//2))
			result[i % n_cl, index] = (norm(examples[i, :, :, 0] - examples[((j*n_cl)+(i%n_cl)), :, :, 0]))
	
	similar_arr = zeros((n_cl, ex_cl), dtype=bool)
	for cl in range(n_cl):
		for ex in range(ex_cl):
			for i in range(ex_cl-ex-1):
				if (result[cl, ed_cl-n_hs(ex_cl-ex) + i] < rtp_mode_collapse_lim):
					similar_arr[cl, ex] = True
					similar_arr[cl, ex+i+1] = True
				#print("cl:%d\tex:%d\ti:%d\trow:%d" % (cl, ex, i, ex+i+1))

	return similar_arr
	"""
	# write to file
	result = transpose(result)
	with open(rtp_folder_name + 'eucl_data_%d.txt' % epoch, 'w') as eucl_file:
		for row in result:
			row_str = ""
			for i, val in enumerate(row):
				if (i > 0):
					row_str += ','
				row_str += '%f' % val
			eucl_file.write(row_str + "\n")
	"""
	"""
	# EUCLIDEAN BOX PLOTTING CODE 
	#eucl_fig = pyplot.figure(figsize=(12, 5), ylim=0, xlabel="class", ylabel="euclidean distance", tight_layout=True)
	eucl_fig, ax = pyplot.subplots(figsize=(12, 5))
	ax.boxplot(transpose(result))
	#ax.set_ylim(bottom=0)
	ax.set_xlabel("class")
	ax.set_ylabel("euclidean distance")
	#pyplot.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
	eucl_fig.savefig(rtp_folder_name + 'euclid_plot_%d.png' % epoch)
	"""

# create and save a plot of generated images
def save_plot(examples, epoch, rows, cols, red_arr):
	fig = pyplot.figure(figsize=(cols * (28/100), rows * (28/100)))
	# plot images
	for i in range(rows * cols):
		# define subplot
		pyplot.subplot(rows, cols, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		cmap = 'gray_r'
		if (red_arr[i%cols, i//cols]):
			cmap = 'Reds'
		pyplot.imshow(examples[i, :, :, 0], cmap=cmap)
	pyplot.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
	fig.savefig(rtp_folder_name + 'out_%d.png' % epoch)
	pyplot.close(fig)
	pyplot.close('all') # to be save


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=rtp_n_classes):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, rtp_conf_list[rtp_list_index]['d_embedding'])(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	#print(merge.shape)
	# downsample
	fe = Conv2D(rtp_conf_list[rtp_list_index]['d_conv_filters'], (3,3), strides=(1,1), padding='same')(merge)
	fe = LeakyReLU(alpha=rtp_conf_list[rtp_list_index]['d_LeReLU_alpha'])(fe)
	fe = Conv2D(rtp_conf_list[rtp_list_index]['d_conv_filters'], (3,3), strides=(2,2), padding='same')(fe)
	#print(fe.shape)
	fe = LeakyReLU(alpha=rtp_conf_list[rtp_list_index]['d_LeReLU_alpha'])(fe)
	#print(fe.shape)
	# downsample
	fe = Conv2D(rtp_conf_list[rtp_list_index]['d_conv_filters'], (3,3), strides=(1,1), padding='same')(fe)
	fe = LeakyReLU(alpha=rtp_conf_list[rtp_list_index]['d_LeReLU_alpha'])(fe)
	fe = Conv2D(rtp_conf_list[rtp_list_index]['d_conv_filters'], (3,3), strides=(2,2), padding='same')(fe)
	#print(fe.shape)
	fe = LeakyReLU(alpha=rtp_conf_list[rtp_list_index]['d_LeReLU_alpha'])(fe)
	#print(fe.shape)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	for i in range(rtp_conf_list[rtp_list_index]['d_hidden_layers1']):
		fe = Dense(rtp_conf_list[rtp_list_index]['d_hidden_units1'])(fe)
		fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	if (rtp_conf_list[rtp_list_index]['SGD'] == 'y'):
		opt = SGD(learning_rate=rtp_conf_list[rtp_list_index]['learn_rate'], momentum=rtp_conf_list[rtp_list_index]['SGD_momentum'], nesterov=(rtp_conf_list[rtp_list_index]['SGD_nesterov']=='y'))
	else:
		opt = Adam(learning_rate=rtp_conf_list[rtp_list_index]['learn_rate'], beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=rtp_n_classes):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, rtp_conf_list[rtp_list_index]['g_embedding'])(in_label)
	# linear multiplication
	n_nodes = 7 * 7
	for i in range(rtp_conf_list[rtp_list_index]['g_hidden_layers1']):
		li = Dense(n_nodes)(li)
		li = LeakyReLU(alpha=rtp_conf_list[rtp_list_index]['g_LeReLU_alpha'])(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = rtp_conf_list[rtp_list_index]['g_hidden_units_mult2'] * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=rtp_conf_list[rtp_list_index]['g_LeReLU_alpha'])(gen)
	for i in range(rtp_conf_list[rtp_list_index]['g_hidden_layers2']-1):
		gen = Dense(n_nodes)(gen)
		gen = LeakyReLU(alpha=rtp_conf_list[rtp_list_index]['g_LeReLU_alpha'])(gen)
	gen = Reshape((7, 7, rtp_conf_list[rtp_list_index]['g_hidden_units_mult2']))(gen)
	
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(rtp_conf_list[rtp_list_index]['g_deconv_filters'], (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=rtp_conf_list[rtp_list_index]['g_LeReLU_alpha'])(gen)
	#print(gen.shape)
	# upsample to 28x28
	gen = Conv2DTranspose(rtp_conf_list[rtp_list_index]['g_deconv_filters'], (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=rtp_conf_list[rtp_list_index]['g_LeReLU_alpha'])(gen)
	#print(gen.shape)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	#print(out_layer.shape)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	if (rtp_conf_list[rtp_list_index]['SGD'] == 'y'):
		opt = SGD(learning_rate=rtp_conf_list[rtp_list_index]['learn_rate'], momentum=rtp_conf_list[rtp_list_index]['SGD_momentum'], nesterov=(rtp_conf_list[rtp_list_index]['SGD_nesterov']=='y'))
	else:
		opt = Adam(learning_rate=rtp_conf_list[rtp_list_index]['learn_rate'], beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load fashion mnist images
def load_real_samples():
	# load dataset
	#(trainX, trainY), (_, _) = load_data()
	
	# load emnist dataset
	(trainX, trainY) = mndata.load_training()
	trainX = asarray(trainX)
	trainY = asarray(trainY)
	trainX = reshape(trainX, [-1, 28, 28])

	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return [X, trainY]

# # select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=rtp_n_classes):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, fid_model, n_epochs=rtp_train_n_epochs, n_batch=rtp_train_n_batch):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	fid_per_epo = 1 	# do not change this
	fid_mod = int(round(bat_per_epo/fid_per_epo))
	fid = 0

	# manually enumerate epochs
	for i in range(n_epochs):
		# create/empty fid-sampling arrays
		fid_samples_fake = zeros((rtp_fid_samples, half_batch, 28, 28, 1))
		fid_samples_fake = fid_samples_fake.astype('float32')
		fid_samples_real = zeros((rtp_fid_samples, half_batch, 28, 28, 1))
		fid_samples_real = fid_samples_real.astype('float32')
		fid_i_count = 0

		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, acc1 = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, acc2 = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)

			if (j % fid_mod >= fid_mod-rtp_fid_samples):
				# save images 10 times every epoch
				fid_samples_fake[fid_i_count] = numpy.asarray([X_fake])
				fid_samples_real[fid_i_count] = numpy.asarray([X_real])
				fid_i_count += 1
			
			# Write to file
			if (j % bat_per_epo == 0):
				file_str = '%.3f, %.3f, %.3f, %.3f, %.3f, %.03f\n' % (d_loss1, d_loss2, g_loss, acc1, acc2, fid)
			else:
				file_str = '%.3f, %.3f, %.3f, %.3f, %.3f\n' % (d_loss1, d_loss2, g_loss, acc1, acc2)
			with open(rtp_folder_name + 'results_csv.txt', 'a') as file:
				file.write(file_str)

			# summarize loss on this batch
			print('>E %d B %d/%d\td_loss_real=%.3f\td_loss_fake=%.3f\tg_loss=%.3f\td_acc_real=%.3f\td_acc_fake=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss, acc1, acc2))

		# Calculate FID after 1 epoch
		fid_samples_fake = reshape(fid_samples_fake, [rtp_fid_samples*half_batch, 28, 28])
		fid_samples_real = reshape(fid_samples_real, [rtp_fid_samples*half_batch, 28, 28])
		
		# convert integer to floating point values
		fid_samples_fake = fid_samples_fake.astype('float32')
		fid_samples_real = fid_samples_real.astype('float32')
		# resize images
		fid_samples_fake = scale_images(fid_samples_fake, (299,299,3))
		fid_samples_real = scale_images(fid_samples_real, (299,299,3))
		# pre-process images
		fid_samples_fake = preprocess_input(fid_samples_fake)
		fid_samples_real = preprocess_input(fid_samples_real)
		# calculate fid
		print("calculating FID with sample size: n_real: %d, n_fake: %d" % (fid_samples_real.shape[0], fid_samples_fake.shape[0]))
		fid = calculate_fid(fid_model, fid_samples_fake, fid_samples_real)
		print(" ->->-> FID for epoch %d: %.03f" % (i + 1, fid))

		img_ex_count = 10
		[img_lat_pnt, img_lbl] = generate_latent_points(latent_dim, img_ex_count*rtp_n_classes)
		for j, _ in enumerate(img_lbl):
			img_lbl[j] = j % rtp_n_classes

		out = g_model.predict([img_lat_pnt, img_lbl])
		out = (out + 1) / 2.0
		similar_char_arr = calc_euclidean_distance(out, rtp_n_classes, (i+1))
		save_plot(out, (i + 1), img_ex_count, rtp_n_classes, similar_char_arr)

		# save the generator model
		f_name = '%d.h5' % (i + 1)
		g_model.save(rtp_folder_name + f_name)

# prepare the inception v3 model
fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
# load image data
dataset = load_real_samples()


for i,conf in enumerate(rtp_conf_list):
	rtp_list_index = i
	rtp_folder_name = rtp_root_folder + str(rtp_list_index) + '/'
	os.makedirs(rtp_folder_name)
	# create the discriminator
	d_model = define_discriminator()
	# create the generator
	g_model = define_generator(n_latent_dim)
	# create the gan
	gan_model = define_gan(g_model, d_model)

	# Change working directory
	#os.chdir(rtp_folder_name + '/') # does not work with emnist

	plot_model(d_model, to_file=(rtp_folder_name + 'd_model.png'), show_shapes=True)
	plot_model(g_model, to_file=(rtp_folder_name + 'g_model.png'), show_shapes=True)

	if (visualize):
		d_model.summary()
		input('\npress enter...')
		g_model.summary()
		input("\npress enter...")
		quit()

	#overwrite csv files
	with open(rtp_folder_name + 'results_csv.txt', 'w') as file:
		file.write('')

	# train model
	train(g_model, d_model, gan_model, dataset, n_latent_dim, fid_model)
