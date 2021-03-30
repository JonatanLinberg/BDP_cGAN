# Use Tkinter for python 2, tkinter for python 3
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import imshow
from matplotlib.pyplot import axis
from matplotlib.pyplot import subplots
from matplotlib.pyplot import close as plt_close
from numpy.random import randn
from numpy import zeros
from numpy import array
from functools import partial
from sys import argv
import os
from math import ceil
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from tensorflow.keras.models import load_model
import tensorflow as tf

# generate points in latent space as input for the generator
def generate_latent_point(latent_dim):
	# generate points in the latent space
	x_input = randn(latent_dim)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(1, latent_dim).astype('float32')
	return z_input


def generate_figure(model, latent_point, class_label):
	plt_close()
	label = zeros((1,), dtype=int)
	label[0] = class_label
	out = model.predict([latent_point, label])
	out = (out + 1) / 2.0
	fig, ax = subplots(figsize=(2.8, 2.8))
	ax.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
	ax = imshow(out[0, :, :, 0], cmap='gray_r')
	return fig 

if (len(argv) > 1):
	f_name = argv[1]
else:
	f_name = input("Enter generator model (.h5) file: ")
try:
	model = load_model(f_name)
except:
	print("Cannot read model file")
	quit()

fig_update_interval = 350
n_latent_dim = model.layers[1].input_shape[0][1] # I think this *always* works
n_slider_frames = 5
col_w = 300
col_h = 1000
lat_scale = 1000
NoneType = type(None)

class GuiGen(tk.Frame):
	def __init__(self, parent):
		self.parent = parent
		self.frame = tk.Frame(parent)
		self.frame.pack(side="top", fill="both", expand=True)
		self.lat_pt = zeros((1, n_latent_dim), dtype='float32')
		self.char_class = 0
		self.should_update_figure = False
		self.slider_frames = []
		self.sliders = []
		self.class_slider = None
		self.n_classes = tk.IntVar()
		self.n_classes.set(47)
		self.saved_vector = zeros((1, n_latent_dim), dtype='float32')
		self.base_vector = zeros((1, n_latent_dim), dtype='float32')
		self.createGUI()

	def createGUI(self):
		# create slider frames and sliders
		for i in range(n_slider_frames):
			self.slider_frames.append(tk.Frame(self.frame, width=col_w, height=col_h))
			self.slider_frames[i].pack_propagate(0)
			self.slider_frames[i].pack(side='left')

		n_rows = ceil(n_latent_dim/n_slider_frames)
		for i, dim in enumerate(self.lat_pt[0]):
			c_delta = i % n_rows
			c_r = int(127 + 64 * ((c_delta%3))) % 256
			c_g = int(127 + 64 * ((c_delta+1)%3)) % 256
			c_b = int(127 + 64 * ((c_delta+2)%3)) % 256
			c_str = '{0:06x}'.format(c_r*(16**4) + c_g*(16**2) + c_b)
			self.sliders.append(tk.Scale(self.slider_frames[i//n_rows], bg='#'+c_str, from_=-3 * lat_scale, to=3 * lat_scale, length=col_w, repeatdelay=200, repeatinterval=1, orient=tk.HORIZONTAL, command=partial(self.update_latent_var, i)))
			self.set_slider_val(i, dim * lat_scale)
			self.sliders[i].pack()

		self.char_frame = tk.Frame(self.frame, width=col_w, height=col_h)
		self.char_frame.pack_propagate(0)
		self.char_frame.pack(side='left')
		self.n_classes.trace_add('write', self.create_class_slider)
		nc_label = tk.Label(self.char_frame, text='n_classes:', width=col_w//2)
		nc_label.place(relx=0.4, rely=0.45, anchor='c')
		class_bound_field = tk.Entry(self.char_frame, textvariable=self.n_classes, width=3)
		class_bound_field.place(relx=0.7, rely=0.45, anchor='c')
		self.create_class_slider()
		self.randomBound = tk.Scale(self.char_frame, from_=0, to=3*lat_scale, label='Random Bound', length=col_w, orient=tk.HORIZONTAL)
		self.randomBound.set(3*lat_scale)
		self.randomBound.place(relx=0.5, rely=0.30, anchor='c')
		randomiseBtn = tk.Button(self.char_frame, text='Randomise', command=self.randomise_latent_point)
		randomiseBtn.place(relx=0.5, rely=0.25, anchor='c')
		normaliseBtn = tk.Button(self.char_frame, text='Normalise', command=self.normalise_latent_point)
		normaliseBtn.place(relx=0.5, rely=0.2, anchor='c')
		saveBtn = tk.Button(self.char_frame, text='Save Vector', command=self.save_vector)
		saveBtn.place(relx=0.5, rely=0.10, anchor='c')
		saveBtn = tk.Button(self.char_frame, text='Save Base Vector', command=self.save_base_vector)
		saveBtn.place(relx=0.5, rely=0.05, anchor='c')
		self.vectorSlider = tk.Scale(self.char_frame, from_=-500, to=500, label='Traverse Vector', length=col_w, orient=tk.HORIZONTAL, command=self.set_vector_percent)
		self.vectorSlider.place(relx=0.5, rely=0.15, anchor='c')
		self.figureFrame = tk.Frame(self.char_frame)
		self.set_should_update_figure(True)
		self.parent.after(0, self.updateFigure)

	def save_base_vector(self):
		self.base_vector = array(self.lat_pt)
		self.vectorSlider.set(0)
		self.update_by_vector(0)

	def save_vector(self):
		self.saved_vector = array(self.lat_pt)
		self.vectorSlider.set(100)
		self.update_by_vector(100)

	def set_vector_percent(self, val):
		self.update_by_vector(int(val))

	def update_by_vector(self, new_percent):
		new_lat_pt = self.base_vector + self.saved_vector * (new_percent / 100)
		for i, dim in enumerate(new_lat_pt[0]):
			self.set_slider_val(i, dim*lat_scale)

	def randomise_latent_point(self):
		new_lat_pt = generate_latent_point(n_latent_dim) * (int(self.randomBound.get())/(3*lat_scale))
		for i, dim in enumerate(new_lat_pt[0]):
			self.set_slider_val(i, dim*lat_scale)

	def normalise_latent_point(self):
		for i in range(n_latent_dim):
			self.set_slider_val(i, 0)

	def set_slider_val(self, i, new_val):
		self.sliders[i].set(new_val)
		self.update_latent_var(i, new_val) #called from slider set event

	def update_latent_var(self, sliderID, new_val):
		self.lat_pt[0, sliderID] = int(new_val)/lat_scale
		self.set_should_update_figure(True)

	def update_char_class(self, new_class):
		self.char_class = int(new_class)
		self.set_should_update_figure(True)

	def updateFigure(self):
		if (self.should_update_figure):
			self.set_should_update_figure(False)
			self.figureFrame.destroy()
			self.figureFrame = tk.Frame(self.char_frame)
			self.figureFrame.place(relx=0.5, rely=0.75, anchor='c')
			tk_fig = FigureCanvasTkAgg(generate_figure(model, self.lat_pt, self.char_class), self.figureFrame)
			tk_fig.get_tk_widget().pack()
		self.parent.after(fig_update_interval, self.updateFigure)

	def create_class_slider(self, *args):
		if (type(self.class_slider) is not NoneType):
			self.class_slider.destroy()
		try:
			self.class_slider = tk.Scale(self.char_frame, repeatdelay=1000, repeatinterval=1000, from_=0, to=self.n_classes.get()-1, length=col_w, label="Class ID",orient=tk.HORIZONTAL, command=self.update_char_class)
			self.class_slider.place(relx=0.5, rely=0.4, anchor='c')
		except Exception:
			pass
		
	def set_should_update_figure(self, new_should_update_figure):
		self.should_update_figure = new_should_update_figure

	def mainloop(self):
		self.parent.mainloop()


        

if __name__ == "__main__":
	root = tk.Tk()
	GuiGen(root).mainloop()
