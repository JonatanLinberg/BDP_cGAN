import os

python_interpreter = "python3"
training_program = "train_cgan.py"
duplicates = 10
exmo_folder = "final_exmos"
exmo_count = 29
exmo_paths = []
commands = []

# create exmo paths
for i in range(exmo_count):
	exmo_paths.append(exmo_folder + "/exmo" + str(i) + ".txt")

# create full command
for i in range(exmo_count):
	command = python_interpreter + " " + training_program + " exmo" + str(i)
	for j in range(duplicates):
		command = command + " " + exmo_paths[i]
	command = command + " &"
	commands.append(command)

for c in commands:
	#print(c)
	os.system(c)
