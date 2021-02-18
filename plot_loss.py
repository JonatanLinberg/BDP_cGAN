import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

if (len(argv) > 1):
	file = argv[1]
else:
	file = input('CSV file name: ')

df = pd.read_csv(file, sep=",", header=None, names=['d_loss1','d_loss2','g_loss'])

df = df.rolling(int(input("Enter window size: "))).mean()

df.plot()
plt.ylim(top=float(input("enter y-limit: ")))

plt.ylim(bottom=0)
plt.show()