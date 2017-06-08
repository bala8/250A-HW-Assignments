# Name: Balachander Padmanabha

import numpy as np
import math
import pprint
import sys

test_list_3 = []
test_list_5 = []
count_3 = 0
count_5 = 0

# weight vector
w = np.zeros((1,64))

# global gradient variable
gradient = np.zeros((64, 1))

# global hessian matrix
hessian = np.zeros((64,64))

def weight_setup():
	global w
	w = np.random.randint(1, size=(1, 64))

def weight_update():
	global w, hessian, gradient
	w = w - np.dot(np.transpose(gradient), np.linalg.inv(hessian))

def hessian_func():
	global test_list_3, test_list_5, w, hessian, count_3, count_5
	
	for i in range(0, 64):
		for j in range(0, 64):
			hessian[i][j] = 0
			for a in range(0, count_3):
				temp = np.dot(w, test_list_3[a])

				sigma_temp = 1 / (1 + math.exp(-temp.item(0,0)))
				sigma_temp_t = 1 / (1 + math.exp(temp.item(0,0)))
				y = 1 - sigma_temp
				y_t = 1 - sigma_temp_t
				hessian[i][j] = hessian[i][j] + (y * y_t * test_list_3[a].item(j,0) * test_list_3[a].item(i,0))
			# samples of 5
			for b in range(0, count_5):
				temp = np.dot(w, test_list_5[b])

				sigma_temp = 1 / (1 + math.exp(-temp.item(0,0)))
				sigma_temp_t = 1 / (1 + math.exp(temp.item(0,0)))

				y = 1 - sigma_temp
				y_t = 1 - sigma_temp_t
				hessian[i][j] = hessian[i][j] + (y * y_t * test_list_5[b].item(j,0) * test_list_5[b].item(i,0))
			hessian[i][j] = -1 * hessian[i][j]
	#pprint.pprint(hessian)

def gradient_func():
	global gradient, test_list_3, test_list_5, w, count_3, count_5
	gradient.fill(0.0)
	# samples of 3
	for i in range(0, count_3):
		temp = np.dot(w, test_list_3[i])
		sigma_temp = 1 / (1 + math.exp(-temp[0][0]))
		y = 1 - sigma_temp
		gradient = gradient + np.dot(y, test_list_3[i])

	# samples of 5
	for j in range(0, count_5):
		temp = np.dot(w, test_list_5[j])
		sigma_temp = 1 / (1 + math.exp(-temp[0][0]))
		y = 0 - sigma_temp
		gradient = gradient  + np.dot(y, test_list_5[j])
	#pprint.pprint(gradient)
	
if __name__ == "__main__":

	with open("train3.txt", "r") as prices:
		for i, line in enumerate(prices):
			line = line.rstrip("\n")
			line = line.rstrip()
			d = line.split(" ")
			# convert string to int while storing in matrix
			d = [int(numeric_string) for numeric_string in d] 
			t = np.array(d).reshape(64,1)
			test_list_3.append(t)
		count_3 = i+1

	with open("train5.txt", "r") as prices:
		for j, line in enumerate(prices):
			line = line.rstrip("\n")
			line = line.rstrip()
			d = line.split(" ")
			# convert string to int while storing in matrix
			d = [int(numeric_string) for numeric_string in d] 
			t = np.array(d).reshape(64,1)
			test_list_5.append(t)
		count_5 = j+1

	weight_setup()
	hessian.fill(0)

	print("\nCaluculating the weight vector, Hessian and Log-likelihood values...")
	log_likelihood = 0.0
	
	# samples of 3
	for i in range(0, count_3):
		temp = np.dot(w, test_list_3[i])
		sigma_t1 = 1 / (1 + math.exp(-temp[0][0]))
		sigma_t2 = 1 / (1 + math.exp(temp[0][0]))
		y = (1 * math.log(sigma_t1)) + ((1-1) * math.log(sigma_t2))
		log_likelihood = log_likelihood + y

	# samples of 5
	for j in range(0, count_5):
		temp = np.dot(w, test_list_5[j])
		sigma_t1 = 1 / (1 + math.exp(-temp[0][0]))
		sigma_t2 = 1 / (1 + math.exp(temp[0][0]))
		y = (0 * math.log(sigma_t1)) + ((1-0) * math.log(sigma_t2))
		log_likelihood = log_likelihood + y

	print("Iteration: 0")
	print("Log-Likelihood value: " + str(log_likelihood))
	print("")
	
	for i in range(0, 100):
		print("Iteration: " + str(i+1))
		gradient_func()
		hessian_func()
		w_temp = w
		weight_update()

		log_likelihood = 0.0
		
		# samples of 3
		for i in range(0, count_3):
			temp = np.dot(w, test_list_3[i])
			sigma_t1 = 1 / (1 + math.exp(-temp[0][0]))
			sigma_t2 = 1 / (1 + math.exp(temp[0][0]))
			y = (1 * math.log(sigma_t1)) + ((1-1) * math.log(sigma_t2))
			log_likelihood = log_likelihood + y

		# samples of 5
		for j in range(0, count_5):
			temp = np.dot(w, test_list_5[j])
			sigma_t1 = 1 / (1 + math.exp(-temp[0][0]))
			sigma_t2 = 1 / (1 + math.exp(temp[0][0]))
			y = (0 * math.log(sigma_t1)) + ((1-0) * math.log(sigma_t2))
			log_likelihood = log_likelihood + y
	
		print("Log-Likelihood value: " + str(log_likelihood))

		hessian.fill(0)
		if(np.allclose(w_temp, w)):
			print("\nFinal weight vector:")
			pprint.pprint(w.reshape(8,8))
			print("\n")
			break
		print("")

	# now to test the model
	i, j = 0.0, 0.0
	count3_train, count5_train = 0.0, 0.0
	mat_3 = []
	mat_5 = []
	print("For Test Data:")
	with open("test3.txt", "r") as prices:
		for i, line in enumerate(prices):
			line = line.rstrip("\n")
			line = line.rstrip()
			d = line.split(" ")
			# convert string to int while storing in matrix
			d = [int(numeric_string) for numeric_string in d] 
			t = np.array(d).reshape(64,1)
			temp = np.dot(w, t)
			sigma_temp = 1/(1+math.exp(-temp[0]))
			if(sigma_temp >= 0.5):
				count3_train = count3_train + 1
				mat_3.append(3)
			else:
				mat_3.append(5)
		i = i + 1

	res = (1 - (count3_train/i)) * 100
	print("\nPercent Error Rate (for images of 3): " + str(res) + "%")
	print("Below is labels of images as per derived model (for test3.txt):")
	print(mat_3)

	with open("test5.txt", "r") as prices:
		for j, line in enumerate(prices):
			line = line.rstrip("\n")
			line = line.rstrip()
			d = line.split(" ")
			# convert string to int while storing in matrix
			d = [int(numeric_string) for numeric_string in d] 
			t = np.array(d).reshape(64,1)
			temp = np.dot(w, t)
			sigma_temp = 1/(1+math.exp(-temp[0]))
			if(sigma_temp < 0.5):
				count5_train = count5_train + 1
				mat_5.append(5)
			else:
				mat_5.append(3)
		j = j+1

	res = (1 - (count5_train/j)) * 100
	print("\nPercent Error Rate (for images of 5): " + str(res) + "%")
	print("Below is labels of images as per derived model (for test5.txt):")
	print(mat_5)

	res = (1 - (count3_train + count5_train) / (i+j)) * 100
	print("Total accuracy is: " + str(res) + "%")
	print("\n")


	# now to test the model
	i, j = 0.0, 0.0
	count3_train, count5_train = 0.0, 0.0
	mat_3 = []
	mat_5 = []
	print("For Train Data")
	with open("train3.txt", "r") as prices:
		for i, line in enumerate(prices):
			line = line.rstrip("\n")
			line = line.rstrip()
			d = line.split(" ")
			# convert string to int while storing in matrix
			d = [int(numeric_string) for numeric_string in d] 
			t = np.array(d).reshape(64,1)
			temp = np.dot(w, t)
			sigma_temp = 1/(1+math.exp(-temp[0]))
			if(sigma_temp >= 0.5):
				count3_train = count3_train + 1
				mat_3.append(3)
			else:
				mat_3.append(5)
		i = i + 1

	res = (1 - (count3_train/i)) * 100
	print("\nPercent Error Rate (for images of 3): " + str(res) + "%")
	print("Below is labels of images as per derived model (for test3.txt):")
	
	with open("train5.txt", "r") as prices:
		for j, line in enumerate(prices):
			line = line.rstrip("\n")
			line = line.rstrip()
			d = line.split(" ")
			# convert string to int while storing in matrix
			d = [int(numeric_string) for numeric_string in d] 
			t = np.array(d).reshape(64,1)
			temp = np.dot(w, t)
			sigma_temp = 1/(1+math.exp(-temp[0]))
			if(sigma_temp < 0.5):
				count5_train = count5_train + 1
				mat_5.append(5)
			else:
				mat_5.append(3)
		j = j+1

	res = (1 - (count5_train/j)) * 100
	print("\nPercent Error Rate (for images of 5): " + str(res) + "%")
	print("Below is labels of images as per derived model (for test5.txt):")
	print("\n")
