import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as fig_to_Tk
from numpy.random import randn
from functools import partial

# generate points in latent space as input for the generator
def generate_latent_point(latent_dim):
	# generate points in the latent space
	x_input = randn(latent_dim)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(1, latent_dim)
	return z_input

def update_char(i, new_val):
	lat_dim[0][i] = new_val

lat_dim = 100
n_cols = 4

root = tk.Tk()

lat_pt = generate_latent_point(lat_dim)
cols = []
for i in range(n_cols):
	cols.append(tk.Frame(root))
	cols[i].pack(side='left')


for i, dim in enumerate(lat_pt[0]):
	slider = tk.Scale(cols[i // (lat_dim//n_cols)], from_=-3, to=3, resolution=0.001, length=300, repeatdelay=300, orient=tk.HORIZONTAL, command=partial(update_char, i))
	slider.set(dim)
	slider.pack()

root.mainloop()