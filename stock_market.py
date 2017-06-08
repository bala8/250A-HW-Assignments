# Name: Balachander Padmanabha

import numpy as np
import pprint
import math
from numpy.linalg import inv

number_sample = 0
sample_list = []
sample_list_2001 = []
number_sample_2001 = 0

result_A = np.arange(16).reshape(4,4)
result_B = np.arange(4).reshape(4,1)
a1, a2, a3, a4 = 0.0, 0.0, 0.0, 0.0
a1_t, a2_t, a3_t, a4_t = 0.0, 0.0, 0.0, 0.0

def linear_coefficients():
	global result_A, result_B, a1, a2, a3, a4
	result_A.fill(0.0)
	result_B.fill(0.0)
	
	for i in range(4, number_sample):
		A = np.array([[sample_list[i-4]], [sample_list[i-3]], [sample_list[i-2]], [sample_list[i-1]]]).reshape(4,1)
		A_t = np.array([sample_list[i-4], sample_list[i-3], sample_list[i-2], sample_list[i-1]]).reshape(1,4)
		temp = np.dot(A, A_t).reshape(4,4)
		result_A = np.add(result_A, temp)

	for i in range(4, number_sample):
		B = np.array([[sample_list[i-4]], [sample_list[i-3]], [sample_list[i-2]], [sample_list[i-1]]]).reshape(4,1)
		B_t = np.array([sample_list[i]]).reshape(1,1)
		temp = np.dot(B, B_t)
		result_B = np.add(result_B, temp)

	result = np.dot(inv(result_A),result_B)
	a4 = round((result[0])[0], 6)
	a3 = round((result[1])[0], 6)
	a2 = round((result[2])[0], 6)
	a1 = round((result[3])[0], 7)
	print("\nThe Linear coefficients are (for 2000):")
	print("a1 = " + str(a1))
	print("a2 = " + str(a2))
	print("a3 = " + str(a3))
	print("a4 = " + str(a4) + "\n")

def mean_square_prediction_error():
	result_error = 0.0
	power = 0.0
	c = 0
	for i in range(4, number_sample):
		power = sample_list[i]-(a1 * sample_list[i-1])-(a2 * sample_list[i-2])-(a3 * sample_list[i-3])-(a4 * sample_list[i-4])
		result_error += math.pow(power, 2)
		temp = 0.0
		c = c + 1

	print("Mean Square Error (for 2000): " + str(result_error/c))

	result_error = 0.0
	power = 0.0
	c = 0
	for j in range(4, number_sample_2001):
		power = sample_list_2001[j]-(a1 * sample_list_2001[j-1])-(a2 * sample_list_2001[j-2])-(a3 * sample_list_2001[j-3])-(a4 * sample_list_2001[j-4])
		result_error += math.pow(power, 2)
		c = c + 1

	print("Mean Square Error (for 2001): " + str(result_error/c))
	print("")

if __name__ == "__main__":

	with open("nasdaq00.txt", "r") as prices:
		for i, line in enumerate(prices):
			line = line.rstrip("\n")
			sample_list.append(float(line))

	number_sample = i+1
	
	with open("nasdaq01.txt", "r") as prices:
		for i, line in enumerate(prices):
			line = line.rstrip("\n")
			sample_list_2001.append(float(line))
		 
	number_sample_2001 = i+1

	linear_coefficients()
	mean_square_prediction_error()

