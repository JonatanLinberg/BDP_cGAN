from scipy.stats import ranksums
import csv

n_exmos = 29
exmos = [str(i) for i in range(n_exmos)]
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
for h in headers:
	print(h, end=',')
print('')


for exmo in exmos:
	data = []
	for run in runs:
		data.append(results[exmo+run])
