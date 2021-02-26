# Use Tkinter for python 2, tkinter for python 3
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


def generate_figure(model, latent_point, class_label):
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


n_latent_dim = 100
n_slider_frames = 4
n_classes = 47
col_w = 300
col_h = 1000
red = "#b55"
green = "#4a6"

class GuiGen(tk.Frame):
	def __init__(self, parent):
		self.parent = parent
		self.frame = tk.Frame(parent)
		self.frame.pack(side="top", fill="both", expand=True)
		self.lat_pt = generate_latent_point(n_latent_dim)
		self.char_class = 10
		self.live = False
		self.createGUI()

	def createGUI(self):
		# create slider frames and sliders
		self.slider_frames = []
		for i in range(n_slider_frames):
			self.slider_frames.append(tk.Frame(self.frame, width=col_w, height=col_h))
			self.slider_frames[i].pack(side='left')

		self.slider_vars = []
		for i, dim in enumerate(self.lat_pt[0]):
			c = tk.DoubleVar()
			slider = tk.Scale(self.slider_frames[i//(n_latent_dim//n_slider_frames)], variable=c, from_=-3, to=3, resolution=0.001, length=col_w, repeatdelay=300, orient=tk.HORIZONTAL)
			c.set(dim)
			self.slider_vars.append(c)
			slider.pack()

		self.char_frame = tk.Frame(self.frame, width=col_w, height=col_h)
		self.char_frame.pack(side='left')
		self.class_slider = tk.Scale(self.char_frame, from_=1, to=n_classes, label="Class ID",orient=tk.HORIZONTAL)
		self.class_slider.set(self.char_class)
		self.class_slider.pack()
		self.live_btn = tk.Button(self.char_frame, text='Toggle Live Update', command=self.toggle_live, bg=red, activebackground=red)
		self.live_btn.pack()

	def toggle_live(self):
		self.live = not self.live
		if (self.live):
			self.live_btn['bg'] = self.live_btn['activebackground'] = green
		else:
			self.live_btn['bg'] = self.live_btn['activebackground'] = red


	def slider_test(self, sliderID):
		self.slider_vars[sliderID] = randn(1)
		self.after(1, self.slider_test)

	def mainloop(self):
		self.parent.mainloop()


        

if __name__ == "__main__":
	root = tk.Tk()
	GuiGen(root).mainloop()