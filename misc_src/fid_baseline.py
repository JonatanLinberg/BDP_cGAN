# example of training an conditional gan on the fashion mnist dataset
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy import reshape
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
from skimage.transform import resize
from mnist import MNIST
from sys import argv

mndata = MNIST('./python-mnist/emnist_data')
mndata.gz = True
mndata.select_emnist('balanced')


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

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

# load fashion mnist images
def load_real_samples():
	# load dataset
	#(trainX, trainy), (_, _) = load_data()
	
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


# size of the latent space
#latent_dim = 100
# create the discriminator
#d_model = define_discriminator()
# create the generator

#if (len(argv) > 1):
#	g_model = load_model(argv[1])
#	print ('loading model', argv[1])
#else:
#	g_model = define_generator(latent_dim)
# create the gan
#gan_model = define_gan(g_model, d_model)


#d_model.summary()
#input()
#gan_model.summary()
#input("\npress enter...")
# load image data
dataset = load_real_samples()

#overwrite csv files
#with open('model_loss_csv.txt', 'w') as file:
#	file.write('')

#with open('model_fid_csv.txt', 'w') as file:
#	file.write('')

# prepare the inception v3 model
fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

# train model
#train(g_model, d_model, gan_model, dataset, latent_dim, fid_model, n_epochs=100)
sample_size = int(input("Enter sample size: "))
n_tests = int(input("Enter number of tests: "))
for i in range(n_tests):
	([x1, _], _) = generate_real_samples(dataset, sample_size)
	([x2, _], _) = generate_real_samples(dataset, sample_size)
	x1 = x1.astype('float32')
	x2 = x2.astype('float32')
	x1 = scale_images(x1, (299, 299, 3))
	x2 = scale_images(x2, (299, 299, 3))
	x1 = preprocess_input(x1)
	x2 = preprocess_input(x2)
	fid = calculate_fid(fid_model, x1, x2)
	print("fid%d: %.03f" % (i, fid))
print("avg fid: %.03f" % (fid/n_tests))
