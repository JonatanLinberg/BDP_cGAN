import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as fig_to_Tk
from numpy.random import randn
from functools import partial
from sys import argv
from keras.models import load_model

# generate points in latent space as input for the generator
def generate_latent_point(latent_dim):
	# generate points in the latent space
	x_input = randn(latent_dim)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(1, latent_dim)
	return z_input

def update_fig():
	print('update_fig')

def update_char(i, new_val):
	lat_pt[0, i] = new_val
	if (live):
		update_fig()

def change_class(new_class):
	char_class = new_class
	if (live):
		update_fig()

if (len(argv) > 1):
	f_name = argv[1]
else:
	f_name = input("Enter generator model (.h5) file: ")
model = load_model(f_name)

n_classes = 47
lat_dim = 100
n_slider_frames = 4
lat_pt = generate_latent_point(lat_dim)
char_class = 10
live = False

root = tk.Tk()

slider_frames = []
for i in range(n_slider_frames):
	slider_frames.append(tk.Frame(root))
	slider_frames[i].pack(side='left')


for i, dim in enumerate(lat_pt[0]):
	slider = tk.Scale(slider_frames[i // (lat_dim//n_slider_frames)], from_=-3, to=3, resolution=0.001, length=300, repeatdelay=300, orient=tk.HORIZONTAL, command=partial(update_char, i))
	slider.set(dim)
	slider.pack()

char_frame = tk.Frame(root)
char_frame.pack(side='left')
class_slider = tk.Scale(char_frame, from_=1, to=n_classes, orient=tk.HORIZONTAL, command=change_class)
class_slider.set(char_class)
class_slider.pack()

root.mainloop()