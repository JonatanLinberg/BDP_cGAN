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
import os
from shutil import copyfile
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
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
# Folder and filename
_name = ('0216_run0_bal_512batch')
# Dataset
# Note! Training or testing is set in the load_real_samples function
mndata.select_emnist('balanced')	# 'balanced', 'byclass'...
_n_classes = 47
# Discriminator parameters
_discriminator_n_embedding = 50
# Generator parameters
_generator_n_embedding = 50
# Training parameters
_train_fid = 10
_train_n_batch = 512	# multiple of 16
_train_n_epochs = 100

# Create directory for next run
os.makedirs(_name + '/')
copyfile('ex_cgan.py', _name + '/ex_cgan.py')

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
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# define the standalone discriminator model
def define_discriminator(in_shape=(28, 28, 1), n_classes=_n_classes):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, _discriminator_n_embedding)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1], 1))(li)
    # image input
    in_image = Input(shape=in_shape)
    # concat label as a channel
    merge = Concatenate()([in_image, li])
    # print(merge.shape)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    # print(fe.shape)
    fe = LeakyReLU(alpha=0.2)(fe)
    # print(fe.shape)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    # print(fe.shape)
    fe = LeakyReLU(alpha=0.2)(fe)
    # print(fe.shape)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([in_image, in_label], out_layer)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim, n_classes=_n_classes):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, _generator_n_embedding)(in_label)
    # linear multiplication
    n_nodes = 7 * 7
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((7, 7, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # print(merge.shape)
    # upsample to 14x14
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    # print(gen.shape)
    # upsample to 28x28
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # print(gen.shape)
    # output
    out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)
    # print(out_layer.shape)
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
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# load fashion mnist images
def load_real_samples():
    # load dataset
    # (trainX, trainy), (_, _) = load_data()

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
def generate_latent_points(latent_dim, n_samples, n_classes=_n_classes):
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
def train(g_model, d_model, gan_model, dataset, latent_dim, fid_model, n_epochs=_train_n_epochs, n_batch=_train_n_batch):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)

            if (j % (bat_per_epo // _train_fid) == 0):
                # FID
                # convert integer to floating point values
                images_fake = X_fake.astype('float32')
                images_real = X_real.astype('float32')
                # resize images
                images_fake = scale_images(images_fake, (299, 299, 3))
                images_real = scale_images(images_real, (299, 299, 3))
                # pre-process images
                images_fake = preprocess_input(images_fake)
                images_real = preprocess_input(images_real)
                # calculate fid
                fid = calculate_fid(fid_model, images_fake, images_real)
                with open('model_fid_csv.txt', 'a') as file:
                    file.write('%.3f, %.3f, %.3f, %.3f\n' % (d_loss1, d_loss2, g_loss, fid))
            else:
                fid = -1.0

            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f fid=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss, fid))
            with open('model_loss_csv.txt', 'a') as file:
                file.write('%.3f, %.3f, %.3f\n' % (d_loss1, d_loss2, g_loss))
        # save the generator model
        f_name = _name + '_%d.h5' % (i + 1)
        g_model.save(f_name)


# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator

if (len(argv) > 1):
    g_model = load_model(argv[1])
    print('loading model', argv[1])
else:
    g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)

# d_model.summary()
# input()
# gan_model.summary()
# input("\npress enter...")
# load image data
dataset = load_real_samples()

# Change working directory
os.chdir(_name + '/')

# overwrite csv file
with open('model_loss_csv.txt', 'w') as file:
    file.write('')

with open('model_fid_csv.txt', 'w') as file:
    file.write('')

# prepare the inception v3 model
fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# train model
train(g_model, d_model, gan_model, dataset, latent_dim, fid_model)
