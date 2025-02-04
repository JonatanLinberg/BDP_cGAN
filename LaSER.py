# Use Tkinter for python 2, tkinter for python 3
print(' Latent Space Explorer and Recorder (LaSER)',
    '\n[==========================================]\n',
	'\nLoading', end='', flush=True)

import tkinter as tk
import imageio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import imshow
from matplotlib.pyplot import axis
from matplotlib.pyplot import subplots
from matplotlib.pyplot import subplot
from matplotlib.pyplot import close as plt_close
from numpy.random import randn
from numpy import zeros
from numpy import array
print('.', end='', flush=True)
from functools import partial
from sys import argv
from sys import stderr
import os
from datetime import datetime as dt
from math import ceil
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
print('.', end='', flush=True)
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
print('.')
import tensorflow as tf
print('Starting...')

fig_update_interval = 350
trav_anim_interval = 350
trav_vec_min = -500
trav_vec_max = 500
red = "#EE2211"
green = "#11BB2A"
n_slider_frames = 5
col_w = 300
col_h = 1000
lat_scale = 1000
NoneType = type(None)
trav_anim_figs = []
gif_speed = 0.15

class n_lat_pt_exception(Exception):
	def __init__(self, got, expected):
		self.got = got
		self.expected = expected

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
	out = model([latent_point, label])
	out = (out + 1) / 2.0
	fig, ax = subplots(figsize=(2.8, 2.8))
	ax.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
	ax = imshow(out[0, :, :, 0], cmap='gray_r')
	return fig 

def _generate_figure_alt(model, latent_point, class_label):
	plt_close()
	label = zeros((1,), dtype=int)
	label[0] = class_label
	out = model([latent_point, label])
	out = (out + 1) / 2.0
	fig, ax = subplots(figsize=(2.8, 2.8))
	model = Model(inputs=model.inputs, outputs=model.layers[-3].output)
	out2 = model([latent_point, label])
	rows = 3
	cols = 3
	for i in range(rows*cols):
		ax = subplot(rows, cols, i+1)
		ax.set_xticks([])
		ax.set_yticks([])
		if (i == 0):
			imshow(out[0, :, :, 0], cmap='gray')
		else:
			imshow(out2[0, :, :, i], cmap="gray")
	return fig

# Uncomment the next line to view filter-output
#generate_figure = _generate_figure_alt

def save_trav_anim():
	path = tk.simpledialog.askstring(title="Save as...", prompt="Animation name:")
	try:
		os.mkdir(path)
	except Exception as ex:
		tk.messagebox.showinfo("Cannot save animation!", ex)
		return
	
	gif_path = path + "/" + path + ".gif"
	with imageio.get_writer(gif_path, mode='I', duration=gif_speed) as writer:

		for i, anim_fig in enumerate(trav_anim_figs):
			f_name = path + '/frame%d.png' % i
			anim_fig.savefig(f_name)
			
			image = imageio.imread(f_name)
			writer.append_data(image)
			try:
				os.remove(f_name)
			except:
				pass

	success_str = "Animation has been saved as " + path + "/" + path + ".gif"
	tk.messagebox.showinfo("Success!", success_str)

class GuiGen(tk.Frame):
	def __init__(self, parent, model):
		self.parent = parent
		self.model = model
		self.n_latent_dim = self.model.layers[1].input_shape[0][1] # I think this *always* works
		self.frame = tk.Frame(parent)
		self.frame.pack(side="top", fill="both", expand=True)
		self.lat_pt = zeros((1, self.n_latent_dim), dtype='float32')
		self.char_class = 0
		self.should_update_figure = False
		self.slider_frames = []
		self.sliders = []
		self.class_slider = None
		self.n_classes = tk.IntVar()
		self.n_classes.set(47)
		self.saved_vector = zeros((1, self.n_latent_dim), dtype='float32')
		self.base_vector = zeros((1, self.n_latent_dim), dtype='float32')
		self.trav_anim = False
		self.travAnimBtn = None
		self.trav_step_size = tk.IntVar()
		self.trav_step_size.set(50)
		self.createGUI()

	def createGUI(self):
		# create slider frames and sliders
		for i in range(n_slider_frames):
			self.slider_frames.append(tk.Frame(self.frame, width=col_w, height=col_h))
			self.slider_frames[i].pack_propagate(0)
			self.slider_frames[i].pack(side='left')

		n_rows = ceil(self.n_latent_dim/n_slider_frames)
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
		nc_label.place(relx=0.4, rely=0.55, anchor='c')
		class_bound_field = tk.Entry(self.char_frame, textvariable=self.n_classes, width=3)
		class_bound_field.place(relx=0.7, rely=0.55, anchor='c')
		self.create_class_slider()
		self.randomBound = tk.Scale(self.char_frame, from_=0, to=3*lat_scale, label='Random Bound', length=col_w, orient=tk.HORIZONTAL)
		self.randomBound.set(3*lat_scale)
		self.randomBound.place(relx=0.5, rely=0.4, anchor='c')
		load_lat_pt_btn = tk.Button(self.char_frame, text="Load Latent Point", command=self.load_latent_point)
		load_lat_pt_btn.place(relx=0.5, rely=0.35, anchor='c')
		save_lat_pt_btn = tk.Button(self.char_frame, text="Save Latent Point", command=self.save_latent_point)
		save_lat_pt_btn.place(relx=0.5, rely=0.32, anchor='c')
		randomiseBtn = tk.Button(self.char_frame, text='Randomise', command=self.randomise_latent_point)
		randomiseBtn.place(relx=0.5, rely=0.29, anchor='c')
		normaliseBtn = tk.Button(self.char_frame, text='Normalise', command=self.normalise_latent_point)
		normaliseBtn.place(relx=0.5, rely=0.26, anchor='c')
		self.vectorSlider = tk.Scale(self.char_frame, from_=trav_vec_min, to=trav_vec_max, label='Traverse Vector', length=col_w, orient=tk.HORIZONTAL, command=self.set_vector_percent)
		self.vectorSlider.place(relx=0.5, rely=0.21, anchor='c')
		saveVecBtn = tk.Button(self.char_frame, text='Save Vector', command=self.save_vector)
		saveVecBtn.place(relx=0.5, rely=0.18, anchor='c')
		saveBaseBtn = tk.Button(self.char_frame, text='Save Base Vector', command=self.save_base_vector)
		saveBaseBtn.place(relx=0.5, rely=0.14, anchor='c')
		self.travAnimBtn = tk.Button(self.char_frame, text='Start Vector Animation', fg=red, command=self.animate_traversal)
		self.travAnimBtn.place(relx=0.5, rely=0.11, anchor='c')
		saveTravAnimBtn = tk.Button(self.char_frame, text='Save Vector Animation', command=save_trav_anim)
		saveTravAnimBtn.place(relx=0.5, rely=0.08, anchor='c')
		step_slider = tk.Scale(self.char_frame, label='Animation Step Size', variable=self.trav_step_size, from_=0, to=trav_vec_max, length=col_w, orient=tk.HORIZONTAL)
		step_slider.place(relx=0.5, rely=0.03, anchor='c')
		self.figureFrame = tk.Frame(self.char_frame)
		self.set_should_update_figure(True)
		self.parent.after(0, self.updateFigure)

	def save_latent_point(self):
		try:
			with tk.filedialog.asksaveasfile(title="Save latent point as:", initialdir="./", filetypes=[("Latent Point file", "*.lptf")]) as outfile:
				outfile.write("#" + str(self.n_latent_dim) + "\n")
				for p in self.lat_pt[0]:
					outfile.write(str(p) + "\n")
		except:
			pass

	def load_latent_point(self):
		in_lpt = zeros((1, self.n_latent_dim), dtype='float32')
		with tk.filedialog.askopenfile(title="Load latent point:", initialdir="./", filetypes=[("Latent Point file", "*.lptf")]) as infile:
			i = 0
			for line in infile:
				if (line[0] == '#'):
					try:
						n_lptf = int(line[1:])
						if (n_lptf != self.n_latent_dim):
							raise n_lat_pt_exception(n_lptf, self.n_latent_dim)
					except Exception as ex:
						tk.messagebox.showinfo("Cannot load latent point!", str(ex))
						return
				else:
					for pt in in_lpt:
						pt[i] = float(line)
					i += 1
		for i, dim in enumerate(in_lpt[0]):
			self.set_slider_val(i, dim*lat_scale)


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
		new_lat_pt = generate_latent_point(self.n_latent_dim) * (int(self.randomBound.get())/(3*lat_scale))
		for i, dim in enumerate(new_lat_pt[0]):
			self.set_slider_val(i, dim*lat_scale)

	def normalise_latent_point(self):
		for i in range(self.n_latent_dim):
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

	def animate_traversal(self):
		self.trav_anim = not self.trav_anim
		if (self.trav_anim):
			trav_anim_figs.clear()
			self.travAnimBtn.configure(fg=green, text="Stop Vector Animation")
			self.step_trav_anim()
		else:
			self.travAnimBtn.configure(fg=red, text="Start Vector Animation")

	def step_trav_anim(self):
		if (self.trav_anim):
			vec_val = self.vectorSlider.get() + self.trav_step_size.get()
			if (vec_val <= trav_vec_max):
				self.vectorSlider.set(vec_val)
			else:
				class_val = self.class_slider.get() + 1
				if (class_val <= self.n_classes.get()-1):
					self.class_slider.set(class_val)
				else:
					self.class_slider.set(0)
				self.vectorSlider.set(trav_vec_min)
			c_fig = self._updateFigure() # direct call
			trav_anim_figs.append(c_fig)
			self.parent.after(trav_anim_interval, self.step_trav_anim)

	def updateFigure(self):
		if (self.should_update_figure and not self.trav_anim):
			self.set_should_update_figure(False)
			self._updateFigure()
		self.parent.after(fig_update_interval, self.updateFigure)

	def _updateFigure(self):
		char_fig = generate_figure(self.model, self.lat_pt, self.char_class)
		tempFrame = tk.Frame(self.char_frame)
		tempFrame.place(relx=0.5, rely=0.75, anchor='c')
		tk_fig = FigureCanvasTkAgg(char_fig, tempFrame)
		tk_fig.get_tk_widget().pack()
		
		self.figureFrame.destroy()
		self.figureFrame = tempFrame
		return char_fig # for animation

	def create_class_slider(self, *args):
		if (type(self.class_slider) is not NoneType):
			self.class_slider.destroy()
		try:
			self.class_slider = tk.Scale(self.char_frame, repeatdelay=1000, repeatinterval=1000, from_=0, to=self.n_classes.get()-1, length=col_w, label="Class ID",orient=tk.HORIZONTAL, command=self.update_char_class)
			self.class_slider.place(relx=0.5, rely=0.5, anchor='c')
		except Exception:
			pass
		
	def set_should_update_figure(self, new_should_update_figure):
		self.should_update_figure = new_should_update_figure

	def mainloop(self):
		self.parent.mainloop()


        

if __name__ == "__main__":
	root = tk.Tk()
	root.title("La.S.E.R.")
	root.minsize(col_w*(n_slider_frames+1), col_h)
	if (len(argv) > 1):
		f_name = argv[1]
	else:
		f_name = tk.filedialog.askopenfilename(title="Select Generator File", initialdir="./", filetypes=[("Model file", "*.h5")])
	try:
		model = load_model(f_name)
	except:
		print("Cannot read model file", file=stderr)
		quit()

	GuiGen(root, model).mainloop()
