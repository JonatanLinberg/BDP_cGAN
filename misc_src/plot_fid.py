import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

if (len(argv) > 1):
	file = argv[1]
else:
	file = input('CSV file name: ')

df = pd.read_csv(file, names=['FID'])
df = df.rolling(int(input("Enter window size: "))).mean()
df.plot()
#plt.ylim(top=float(input("enter y-limit: ")))
plt.show()