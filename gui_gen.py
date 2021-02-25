import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as fig_to_tk
from matplotlib import pyplot
from numpy.random import randn
from numpy import zeros
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

def generate_figure(model, latent_point, class_label, frame):
	label = zeros((1,), dtype=int)
	label[0] = class_label
	out = model.predict([latent_point, label])
	out = (out + 1) / 2.0
	fig, ax = pyplot.subplots()
	pyplot.axis('off')
	ax = pyplot.imshow(out[0, :, :, 0], cmap='gray_r')
	return fig 

if (len(argv) > 1):
	f_name = argv[1]
else:
	f_name = input("Enter generator model (.h5) file: ")
model = load_model(f_name)

root = tk.Tk()
n_classes = 47
lat_dim = 100
n_slider_frames = 4
lat_pt = generate_latent_point(lat_dim)
char_class = 10
live = False
col_w = 300
col_h = 700
red = "#b55"
green = "#4a6"

def update_fig():
	pass

def update_latent_dim(i, new_val):
	lat_pt[0, i] = float(new_val.replace(',','.'))
	if (live):
		update_fig()

def change_class(new_class):
	global char_class
	char_class = new_class
	if (live):
		update_fig()

def toggle_live():
	global live
	live = not live
	if (live):
		live_btn['bg'] = live_btn['activebackground'] = green
		update_fig()
	else:
		live_btn['bg'] = live_btn['activebackground'] = red



slider_frames = []
for i in range(n_slider_frames):
	slider_frames.append(tk.Frame(root))
	slider_frames[i].pack(side='left')


for i, dim in enumerate(lat_pt[0]):
	slider = tk.Scale(slider_frames[i // (lat_dim//n_slider_frames)], from_=-3, to=3, resolution=0.001, length=col_w, repeatdelay=300, orient=tk.HORIZONTAL, command=partial(update_latent_dim, i))
	slider.set(dim)
	slider.pack()

char_frame = tk.Frame(root, width=col_w, height=col_h)
char_frame.pack(side='left')
char_frame.pack_propagate(0)
class_slider = tk.Scale(char_frame, from_=1, to=n_classes, label="Class ID",orient=tk.HORIZONTAL, command=change_class)
class_slider.set(char_class)
class_slider.pack()
live_btn = tk.Button(char_frame, text='Toggle Live Update', command=toggle_live, bg=red, activebackground=red)
live_btn.pack()

update_fig()

root.mainloop()