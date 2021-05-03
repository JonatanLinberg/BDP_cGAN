from scipy.stats import mannwhitneyu
import csv

n_exmos = 29
exmos = [str(i) for i in range(1, n_exmos)]
for i, exmo in enumerate(exmos):
	if (len(exmo) < 2):
		exmos[i] = '0' + exmo
runs = (' A', ' B', ' C')
data_cols = [0, 2, 6, 7, 8, 10]

infile = "derived_results_csv.txt"
results = {}

# load data
with open(infile, 'r') as file:
	reader = csv.reader(file)
	results = {row[0]:row[1:] for row in reader}


headers = [results['EXMO'][i] for i in data_cols]
print(', '.join(['EXMO'] + headers))

# baseline 
baseline_data = []
for run in runs:
	baseline_data.append([results['00'+run][i] for i in data_cols])

for exmo in exmos:
	data = []
	for run in runs:
		data.append([results[exmo+run][i] for i in data_cols])
	out = [exmo]
	for i in range(len(data_cols)):
		st, p = mannwhitneyu([data[j][i] for j in range(len(runs))], [baseline_data[j][i] for j in range(len(runs))])
		out += ["%.3f"%p]
	print(', '.join(out))

	#print(','.join([exmo]+data[0]))
