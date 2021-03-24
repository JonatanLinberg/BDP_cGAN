from tensorflow.keras.models import load_model
from numpy import zeros
from numpy import array
from numpy import append
from numpy import reshape
from matplotlib import pyplot as plt
from sys import argv

def plot_imgs(images):
	c_str = 'gray_r'
	fig, ax = plt.subplots(1, images.shape[0])
	for i, _ in enumerate(images):
		ax[i].tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
		if (i == images.shape[0]-1):
			c_str = "Reds"
		ax[i].imshow(images[i, :, :, 0], cmap=c_str)
	#plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
	plt.show()

def create_diff_img(img_a, img_b):
	img_c = zeros((img_a.shape), dtype='float32')
	for i, _ in enumerate(img_a):
		for j, _ in enumerate(img_a[i]):
			img_c[i, j] = max((img_a[i, j] - img_b[i, j]), 0)
	img_c[img_c[:, :, 0] <= 0.05] = None
	return img_c


model = None
images = zeros((3, 28, 28, 1), dtype='float32')
if (len(argv) > 1):
	model = load_model(argv[1])

if (model is not None):
	out = model.predict([[zeros((1, model.layers[1].input_shape[0][1]), dtype='float32')], array([10])])
	out = (out + 1) / 2.0
	images[0] = out
	images[1] = zeros((), dtype='float32')
	images[1, :, 1:] = images[0, :, :-1]
	images[2] = create_diff_img(images[0], images[1])
	plot_imgs(images)