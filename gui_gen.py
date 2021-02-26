# Use Tkinter for python 2, tkinter for python 3
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import imshow
from matplotlib.pyplot import axis
from matplotlib.pyplot import subplots
from matplotlib.pyplot import close as plt_close
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
	plt_close()
	label = zeros((1,), dtype=int)
	label[0] = class_label
	out = model.predict([latent_point, label])
	out = (out + 1) / 2.0
	fig, ax = subplots(figsize=(2.8, 2.8))
	axis('off')
	ax = imshow(out[0, :, :, 0], cmap='gray_r')
	return fig 

if (len(argv) > 1):
	f_name = argv[1]
else:
	f_name = input("Enter generator model (.h5) file: ")
model = load_model(f_name)

fig_update_interval = 300
n_latent_dim = 100
n_slider_frames = 4
n_classes = 47
col_w = 300
col_h = 1050
lat_scale = 1000


class GuiGen(tk.Frame):
	def __init__(self, parent):
		self.parent = parent
		self.frame = tk.Frame(parent)
		self.frame.pack(side="top", fill="both", expand=True)
		self.lat_pt = generate_latent_point(n_latent_dim)
		self.char_class = 10
		self.should_update_figure = False
		self.slider_frames = []
		self.createGUI()

	def createGUI(self):
		# create slider frames and sliders
		for i in range(n_slider_frames):
			self.slider_frames.append(tk.Frame(self.frame, width=col_w, height=col_h))
			self.slider_frames[i].pack_propagate(0)
			self.slider_frames[i].pack(side='left')

		n_rows = (n_latent_dim//n_slider_frames)
		for i, dim in enumerate(self.lat_pt[0]):
			c_delta = i % n_rows
			c_str = '{0:06x}'.format(16777215-int(c_delta*256/n_rows*16**4 + c_delta*256/n_rows*16**2 + c_delta*256/n_rows))
			slider = tk.Scale(self.slider_frames[i//n_rows], bg='#'+c_str, from_=-3 * lat_scale, to=3 * lat_scale, length=col_w, orient=tk.HORIZONTAL, command=partial(self.update_latent_var, i))
			slider.set(dim * lat_scale)
			slider.pack()

		self.char_frame = tk.Frame(self.frame, width=col_w, height=col_h)
		self.char_frame.pack_propagate(0)
		self.char_frame.pack(side='left')
		self.class_slider = tk.Scale(self.char_frame, from_=0, to=n_classes-1, length=col_w, label="Class ID",orient=tk.HORIZONTAL, command=self.update_char_class)
		self.class_slider.set(self.char_class)
		self.class_slider.place(relx=0.5, rely=0.3, anchor='c')
		self.figureFrame = tk.Frame(self.char_frame)
		self.set_should_update_figure(True)
		self.parent.after(0, self.updateFigure)

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
			self.figureFrame.place(relx=0.5, rely=0.5, anchor='c')
			tk_fig = FigureCanvasTkAgg(generate_figure(model, self.lat_pt, self.char_class), self.figureFrame)
			tk_fig.get_tk_widget().pack()
		self.parent.after(fig_update_interval, self.updateFigure)

	def set_should_update_figure(self, new_should_update_figure):
		self.should_update_figure = new_should_update_figure

	def mainloop(self):
		self.parent.mainloop()


        

if __name__ == "__main__":
	root = tk.Tk()
	GuiGen(root).mainloop()