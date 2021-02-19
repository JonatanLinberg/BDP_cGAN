import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

if (len(argv) > 1):
	file = argv[1]
else:
	file = input('CSV file name: ')

df = pd.read_csv(file, names=['d_loss_real','d_loss_fake','g_loss', 'd_acc_real', 'd_acc_fake','FID'])
#print(df.head())
df = df.interpolate()
#print(df.head())
df = df.rolling(int(input("Enter window size: "))).mean()

choice = int(input('1. all plots\n2. loss only\n3. accuracy only\n4. FID only\nChoose: '))
if (choice == 1):
	plot, sub = plt.subplots(3)
	sub[0].plot(df[['d_loss_real','d_loss_fake','g_loss']])
	sub[0].legend(('d_loss_real','d_loss_fake','g_loss'))
	sub[0].set_ylabel('loss')
	sub[1].plot(df[['d_acc_real', 'd_acc_fake']])
	sub[1].legend(('d_acc_real', 'd_acc_fake'))
	sub[1].set_ylabel('accuracy')
	sub[2].plot(df['FID'])
	sub[2].set_ylabel('FID')
	for s in sub:
		s.set_xlabel('number of batches')
		s.set_ylim(bottom=0)
else:
	if (choice == 2):
		df[['d_loss_real','d_loss_fake','g_loss']].plot()
		plt.ylabel('loss')
	elif (choice == 3):
		df[['d_acc_real', 'd_acc_fake']].plot()
		plt.ylabel('accuracy')
	elif (choice == 4):
		df['FID'].plot()
		plt.ylabel('FID')
	plt.xlabel('number of batches')
	plt.ylim(bottom=0)

plt.show()