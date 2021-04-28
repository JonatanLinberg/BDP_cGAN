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
DEBUG = False

for path, name, files in os.walk(experiment_folder):
	for file in files:
		if (file == result_file_name):
			results_in = pd.read_csv(path + "/" + file, names=['d_loss_real','d_loss_fake','g_loss', 'd_acc_real', 'd_acc_fake','FID'])
			bat_fid_1 = -1
			bat_fid_2 = -1
			bat_stab_1 = -1
			bat_stab_2 = -1
			bat_stab_3 = -1
			bat_stab_4 = -1

			# count FID
			for i, val in enumerate(results_in['FID']):
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
			
			if (DEBUG):
				print(path.split("/")[1])
				print("fid_1:", bat_fid_1)
				print("fid_2:", bat_fid_2)
				print("stab_1:", bat_stab_1)
				print("stab_2:", bat_stab_2)
				print("stab_3:", bat_stab_3)
				print("stab_4:", bat_stab_4)
				print("\n")

			open(path + "/calculated_results.txt", 'w') # create file
			with open(path + "/calculated_results.txt", 'a') as results_out:
				results_out.write("fid_1: " + str(bat_fid_1))
				results_out.write("fid_2: " + str(bat_fid_2))
				results_out.write("stab_1: " + str(bat_stab_1))
				results_out.write("stab_2: " + str(bat_stab_2))
				results_out.write("stab_3: " + str(bat_stab_3))
				results_out.write("stab_4: " + str(bat_stab_4))
print("DONE!")
