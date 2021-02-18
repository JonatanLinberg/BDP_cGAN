from mnist import MNIST
import random

mndata = MNIST('./emnist_data')
mndata.gz = True
mndata.select_emnist('byclass')
images, labels = mndata.load_training()

characters = ['0','1','2','3','4','5','6','7','8','9',
			  'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
			  'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

while (True):
	input("\n\n\nPress any key to print a character...\n")
	img_index = random.randint(0, len(images))

	print("Character:",labels[img_index],'(',characters[labels[img_index]],')',"\tID:",img_index)

	for i, pix in enumerate(images[img_index]):
		if (i % 28 == 0):
			print("")
		if (pix > 175):
			print(chr(0x2593), end='')
		elif (pix > 75):
			print(chr(0x2592), end='')
		elif (pix > 0):
			print(chr(0x2591), end='')
		else:
			print(' ', end='')
