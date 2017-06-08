# Name: Balachander Padmanabha

import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
import pdb

start_time = time.time()
T = 180000 # number of Observation
n = 26 # number of hidden state

#initial state transitions list (26x1)
pi = np.zeros((26,1))

#transition matrix (26x26)
A_matrix = np.zeros((26,26))

#emission matrix (26x2)
B_matrix = np.zeros((26,2))

#Observation (T = 180000)
Obs_matrix = np.zeros((1, T))

#final result 
result = np.ones(T)

def read_data():
	print("Reading Data...")
	global Obs_matrix, pi, A_matrix, B_matrix

	f = open("initial_state_distributions.txt", "r")
	pi = np.array([float(i) for i in f.read().strip().split()])
	f.close()

	f = open("observations.txt", "r")
	Obs_matrix = np.array([int(i) for i in f.read().strip().split()])
	f.close()

	f = open("transitionMatrix.txt", "r")
	A_matrix = np.array([[float(i) for i in line.strip().split()] for line in f])
	f.close()
			
	f = open("emissionMatrix.txt", "r")
	B_matrix = np.array([[float(i) for i in line.strip().split()] for line in f])
	f.close()

def viterbi():
	print("Running Viterbi Algorithm...")
	global A_matrix, B_matrix, Obs_matrix, pi, T, n, result
	best_states_prev = np.ones([n, T])
	
	# base case
	li_matrix = np.zeros([n , T])
	li_matrix[:, 0] = np.log(pi) * np.log(B_matrix[:, Obs_matrix[0]])

	for t in range(1, T):
		li_matrix[:, t] = [np.max(li_matrix[:, t-1] + np.log(A_matrix[:, j])) + np.log(B_matrix[j][Obs_matrix[t]]) for j in range(n)]
		best_states_prev[:, t] = [np.argmax(li_matrix[ :, t-1] + np.log(A_matrix[ :, j]) + np.log(B_matrix[j, Obs_matrix[t]])) for j in range(n)]

	pdb.set_trace()
	result = -1 * result
	result[-1] = np.argmax(li_matrix[:, T-1])

	for t in range(T-2, -1, -1):
		result[t-T] = best_states_prev[result[t-T+1], t-T+1]

	char_temp = {0 : 'A', 1 : 'B', 2: 'C', 3 : 'D', 4 : 'E', 5: 'F', 6 : 'G', 
	             7 : 'H', 8: 'I', 9 : 'J', 10 : 'K', 11: 'L', 12 : 'M', 13 : 'N', 
	             14: 'O', 15 : 'P', 16 : 'Q', 17: 'R', 18 : 'S', 19 : 'T', 20: 'U',
	             21 : 'V', 22 : 'W', 23: 'X', 24 : 'Y', 25 : 'Z'}

	pdb.set_trace()
	p = -1
	final_result = ""
	for i in result:
		if(i != p):
			final_result = final_result + str(char_temp[i])
			p = i

	print("=> Final String: " + str(final_result))
	pdb.set_trace()
	print("")

def plot_result():
	print("Plotting Graph...")
	plt.title("7.1 HMM-Encode Plot")
	data_x = []
	data_y = []
	for xt in range(T):
		data_x.append(xt)
		data_y.append(result[xt])

	lines = plt.plot(data_x, data_y)
	plt.ylabel("Alphabet")
	plt.xlabel("Iteration (t)")
	plt.setp(lines, color='r', linewidth=2.0)
	plt.show()

if __name__ == '__main__':
	read_data()
	viterbi()
	plot_result()
	print("=> RUNTIME: %s seconds" % (time.time() - start_time))


