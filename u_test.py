from scipy.stats import mannwhitneyu
import csv
import numpy

n_exmos = 29
exmos = [str(i) for i in range(n_exmos)]
for i, exmo in enumerate(exmos):
	if (len(exmo) < 2):
		exmos[i] = '0' + exmo
runs = (' A', ' B', ' C')
data_cols = [0, 2, 6, 7, 8, 10]
alt_hyp_cols = ['less', 'less', 'greater', 'greater', 'greater', 'greater']

infile = "data/derived_results.csv"
results = {}

# load data
with open(infile, 'r') as file:
	reader = csv.reader(file)
	results = {row[0]:row[1:] for row in reader}


headers = [results['EXMO'][i] for i in data_cols]
print(', '.join(['EXMO'] + headers))

data = []
for i, exmo in enumerate(exmos):
	data.append([[] for j, _ in enumerate(data_cols)])
	for run in runs:
		for j, col in enumerate(data_cols):
			try:
				data[i][j].append(int(results[exmo+run][col]))
			except:
				pass


for i, exmo in enumerate(exmos):
	out = [exmo]
	rest_data = []
	
	for j in range(len(data_cols)):
		rest_data.append([])
		if (data[i][j] != []):
			for d in data[:i] + data[i+1:]:
				rest_data[j] += d[j]

			_, p = mannwhitneyu(data[i][j], rest_data[j], alternative=alt_hyp_cols[j])
			out += ["%.3f"%p]
		else:
			out += ["-----"]
	print(','.join(out))
