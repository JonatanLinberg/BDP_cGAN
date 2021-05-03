from scipy.stats import mannwhitneyu
import csv

n_exmos = 29
exmos = [str(i) for i in range(1, n_exmos)]
for i, exmo in enumerate(exmos):
	if (len(exmo) < 2):
		exmos[i] = '0' + exmo
runs = (' A', ' B', ' C')
data_cols = [0, 2, 6, 7, 8, 10]
alt_hyp_cols = ['less', 'less', 'greater', 'greater', 'greater', 'greater']

infile = "derived_results_csv.txt"
results = {}

# load data
with open(infile, 'r') as file:
	reader = csv.reader(file)
	results = {row[0]:row[1:] for row in reader}


headers = [results['EXMO'][i] for i in data_cols]
print(', '.join(['EXMO'] + headers))

# baseline 
baseline_data = {col:[] for col in data_cols}
for run in runs:
	for i, col in enumerate(data_cols):
		try:
			baseline_data[col].append(int(results['00'+run][col]))
		except:
			pass

for exmo in exmos:
	data = {col:[] for col in data_cols}
	for run in runs:
		for i, col in enumerate(data_cols):
			try:
				data[col].append(int(results[exmo+run][col]))
			except:
				pass

	out = [exmo]
	for i, col in enumerate(data_cols):
		if (data[col] != []):
			_, p = mannwhitneyu(data[col], baseline_data[col], alternative=alt_hyp_cols[i])
			out += ["%.3f"%p]
		else:
			out += ["-----"]
		
	print(', '.join(out))