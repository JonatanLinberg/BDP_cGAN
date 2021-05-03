import pandas as pd
from matplotlib import pyplot as plt
import os

win_size = 25
FID_1 = 5
FID_2 = 2
ACC_LOW_1 = 0.5
ACC_HIGH_1 = 0.85
ACC_LOW_2 = 0.6
ACC_HIGH_2 = 0.75
experiment_folder = "final_experiment"
result_file_name = "results_csv.txt"

modelNum = ('A','B','C','D','E','F')
headers = "EXMO, FID < 5 (batches), FID < 5, FID < 2 (batches), FID < 2, , EXMO, Acc. 50-85%, Acc. 60-75%, Avg. Acc. 50-85%, Avg. Acc. 50-85% (epochs), Avg. Acc. 60-75%, Avg. Acc. 60-75% (epochs), , EXMO, argmin(FID), min(FID)"
print(headers)

def out_str(input):
	if (input == -1):
		return ""
	else:
		return str(input)

for path, name, files in os.walk(experiment_folder):
	for file in files:
		if (file == result_file_name):
			results_in = pd.read_csv(path + "/" + file, names=['d_loss_real','d_loss_fake','g_loss', 'd_acc_real', 'd_acc_fake','FID'])
			bat_fid_1 = -1
			bat_fid_2 = -1
			min_fid = 1000
			min_fid_bat = -1
			bat_stab_1 = 0
			bat_stab_2 = 0
			bat_stab_3 = 0
			bat_stab_4 = 0

			# count FID
			for i, val in enumerate(results_in['FID']):
				if (val > 0 and val < min_fid):
					min_fid = val
					min_fid_bat = i
				if (bat_fid_1 == -1 and val > 0 and val < FID_1):
					bat_fid_1 = i
				if (bat_fid_2 == -1 and val > 0 and val < FID_2):
					bat_fid_2 = i
				if (bat_fid_1 != -1 and bat_fid_2 != -1):
					break
			
			# Stablity
			s1_start = -1
			s2_start = -1
			for i, val in enumerate(results_in['d_acc_fake']):
				if (s1_start == -1):
					if (val > ACC_LOW_1 and val < ACC_HIGH_1):
						s1_start = i
				else:
					if (val < ACC_LOW_1 or val > ACC_HIGH_1):
						new_bat_stab = i - s1_start
						if (bat_stab_1 < new_bat_stab):
							bat_stab_1 = new_bat_stab
						s1_start = -1

				if (s2_start == -1):
					if (val > ACC_LOW_2 and val < ACC_HIGH_2):
						s2_start = i
				else:
					if (val < ACC_LOW_2 or val > ACC_HIGH_2):
						new_bat_stab = i - s2_start
						if (bat_stab_2 < new_bat_stab):
							bat_stab_2 = new_bat_stab
						s2_start = -1

			avg_d_loss = results_in['d_acc_fake'].rolling(win_size).mean()
			s3_start = -1
			s4_start = -1
			for i, val in enumerate(avg_d_loss):
				if (s3_start == -1):
					if (val > ACC_LOW_1 and val < ACC_HIGH_1):
						s3_start = i
				else:
					if (val < ACC_LOW_1 or val > ACC_HIGH_1):
						new_bat_stab = i - s3_start
						if (bat_stab_3 < new_bat_stab):
							bat_stab_3 = new_bat_stab
						s3_start = -1

				if (s4_start == -1):
					if (val > ACC_LOW_2 and val < ACC_HIGH_2):
						s4_start = i
				else:
					if (val < ACC_LOW_2 or val > ACC_HIGH_2):
						new_bat_stab = i - s4_start
						if (bat_stab_4 < new_bat_stab):
							bat_stab_4 = new_bat_stab
						s4_start = -1

			exmo_num = path.split("/")[1][4:]
			if (len(exmo_num) < 2):
				exmo_num = '0' + exmo_num

			exmo_name = exmo_num + ' ' + modelNum[int(path.split("/")[2])]

			out = exmo_name + ', ' + \
					out_str(bat_fid_1) + ', , ' + \
					out_str(bat_fid_2) + ', , , ' + \
					exmo_name + ', ' + \
					out_str(bat_stab_1) + ', ' + \
					out_str(bat_stab_2) + ', ' + \
					out_str(bat_stab_3) + ', , ' + \
					out_str(bat_stab_4) + ', , , ' + \
					exmo_name + ', ' + \
					out_str(min_fid_bat) + ', ' + \
					out_str(min_fid)
			print(out)
