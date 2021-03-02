import pandas as pd
import matplotlib.pyplot as plt
from numpy import arange
from sys import argv

filenames = []
average_p_col = False
df_list = []
x_len = 0
x_steps = 5
col_w = 6
col_h = 8

for i in range(1, len(argv)):
	filenames.append(argv[i])

if (len(filenames) == 0):
	filenames.extend(input("Enter file name(s): ").split())

if (len(filenames) > 1):
	average_p_col = int(input("1. Average results per column\n2. Show as subplots\nChoose: ")) == 1

for file in filenames:
	print("Reading", file, "as CSV file...")
	df = pd.read_csv(file, names=['d_loss_real','d_loss_fake','g_loss', 'd_acc_real', 'd_acc_fake','FID'])
	#print(df.head())
	df = df.interpolate()
	#print(df.head())
	df = df.rolling(int(input("Enter window size: "))).mean()
	df_list.append(df)

if (average_p_col):
	print("Averaging results...")
	df = df_list[0]
	# sum dataframes
	for i in range(1, len(df_list)):
		df += df_list[i]
	df = df / len(df_list)
	df_list = [df]

for df in df_list:
	if (df.shape[0] > x_len):
		x_len = df.shape[0]

if (len(df_list) > 1):
	plot, ax = plt.subplots(len(df_list), 3, figsize=(col_w*len(df_list), col_h))
	for i in range(len(df_list)):
		ax[i, 0].plot(df_list[i][['d_loss_real','d_loss_fake','g_loss']])
		ax[i, 0].legend(('d_loss_real','d_loss_fake','g_loss'))
		ax[i, 0].set_ylabel('loss')
		ax[i, 1].plot(df_list[i][['d_acc_real', 'd_acc_fake']])
		ax[i, 1].legend(('d_acc_real', 'd_acc_fake'))
		ax[i, 1].set_ylabel('accuracy')
		ax[i, 2].plot(df_list[i]['FID'])
		ax[i, 2].set_ylabel('FID')
	for i in range(len(ax)):
		ax[i, 0].set_title(filenames[i], loc='left')
		for a in ax[i]:
			a.set_xlabel('number of batches')
			a.set_xlim(right=x_len)
			a.set_xticks(arange(0, x_len+1, x_len//x_steps, dtype=int))
			a.set_ylim(bottom=0)
else:
	choice = int(input('1. all plots\n2. loss only\n3. accuracy only\n4. FID only\nChoose: '))
	if (choice == 1):
		plot, sub = plt.subplots(3, figsize=(col_w, col_h))
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

plt.subplots_adjust(left=0.08, right=0.92, hspace=0.5, wspace=0.2)
plt.show()