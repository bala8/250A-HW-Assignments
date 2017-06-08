# Name: Balachander Padmanabha

import math
import pprint
import numpy as np
from itertools import repeat

T = 267
n = 23
X_list = []
Y_list = []
p_matrix = np.zeros((T,n))

# list to store final results
result = []

# pi list
pi = []


def denominator(x):
	global pi, n
	res = 1.0
	for i in range(n):
		pi_i = 1 - pi[i]
		xi = x[i]
		res = res * math.pow(pi_i,xi)
	return 1-res

def e_step():
	for i in range(T):
		y_k = Y_list[i]
		x_k = X_list[i]
		sum_deno = denominator(x_k)
		for j in range(n):
			x = x_k[j]
			p = pi[j]
			p_temp = float(float(y_k) * float(x) * float(p))
			p_matrix[i][j] = p_temp/sum_deno

def m_step():
	global pi, n, T
	for i in range(n):
		deno = 0.0
		nume = 0.0 
		for j in range(T):
			if (int(X_list[j][i]) == 1):
				deno += 1
			nume += p_matrix[j][i]
		pi[i] = nume/deno

def log_likelihood_func(ite):
	log_likelihood = 0.0
	miss_count = 0
	for i in range(T):
		val = denominator(X_list[i])
	
		if(((val >= 0.5) & ( int(Y_list[i])==0)) | ((val <= 0.5) & (int(Y_list[i])==1))):
			miss_count+=1
		
		if(Y_list[i] == 1):
			p = val
		else:
			p = 1-val

		log_likelihood += math.log(p)
	
	log_likelihood = log_likelihood / T
	print("Iteration: " + str(ite) + " Log-Likehood: " + str(log_likelihood) + " Mistakes: " + str(miss_count))

if __name__ == '__main__':
	pi = pi + list(repeat((1.0/n), 23))
	
	with open("specX.txt", "r") as xes:
		for i, line in enumerate(xes):
			line = line.rstrip("\n")
			line = line.rstrip()
			temp = line.split(" ")
			temp = [int(numeric_string) for numeric_string in temp]
			X_list.append(temp)
		
	with open("specY.txt", "r") as yes:
		for i, line in enumerate(yes):
			line = line.rstrip("\n")
			line = line.rstrip()
			Y_list.append(int(line))
	
	log_likelihood_func(0)
	for i in range(256):
		e_step()
		m_step()
		log_likelihood_func(i+1)

