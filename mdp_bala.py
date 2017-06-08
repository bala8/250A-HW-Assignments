# Name: Balachander Padmanabha

import random
import copy
import numpy as np
from numpy.linalg import inv

gamma = 0.9925
num_states = 81
num_action = 4
num_iteration = 2000

# initializing
value_matrix = [0] * num_states
policy_matrix = [0] * num_states

prob_a = [[0 for x in range(num_states)] for y in range(num_states)]
prob_b = [[0 for x in range(num_states)] for y in range(num_states)]
prob_c = [[0 for x in range(num_states)] for y in range(num_states)]
prob_d = [[0 for x in range(num_states)] for y in range(num_states)]
rewards = [0] * num_states

def read_data():
	global prob_a, prob_d, prob_c, prob_d, rewards
	print("Reading the data...")

	with open("prob_a1.txt", "r") as lines:
		for i, line in enumerate(lines):
			line = line.rstrip("\n")
			s, s_t, p = line.split()
			prob_a[int(s) - 1][int(s_t) - 1] = float(p)

	with open("prob_a2.txt", "r") as lines:
		for i, line in enumerate(lines):
			line = line.rstrip("\n")
			s, s_t, p = line.split()
			prob_b[int(s) - 1][int(s_t) - 1] = float(p)

	with open("prob_a3.txt", "r") as lines:
		for i, line in enumerate(lines):
			line = line.rstrip("\n")
			s, s_t, p = line.split()
			prob_c[int(s) - 1][int(s_t) - 1] = float(p)

	with open("prob_a4.txt", "r") as lines:
		for i, line in enumerate(lines):
			line = line.rstrip("\n")
			s, s_t, p = line.split()
			prob_d[int(s) - 1][int(s_t) - 1] = float(p)

	with open("rewards.txt" , "r") as lines:
		for i, line in enumerate(lines):
			line = line.rstrip("\n")
			rewards[i] = int(line)

def helper_func(s):
	global value_matrix, prob_a, prob_b, prob_c, prob_d
	max_a = 0
	max_val = -99999
	for a in range(num_action):
		val = 0
		for s_t in range(num_states):
			if(a == 0):
				val = val + (prob_a[s][s_t] * value_matrix[s_t])
			elif(a == 1):
				val = val + (prob_b[s][s_t] * value_matrix[s_t])
			elif(a == 2):
				val = val + (prob_c[s][s_t] * value_matrix[s_t])
			elif(a == 3):
				val = val + (prob_d[s][s_t] * value_matrix[s_t])

		if(val > max_val):
			max_val = val
			max_a = a

	return(max_val, max_a)


def iteration_func():
	global policy_matrix, value_matrix, num_iteration
	print("Iterating through the data...")
	for ite in range(num_iteration):
		temp = copy.copy(value_matrix)
		for state in range(num_states):
			value, idx = helper_func(state)
			value_matrix[state] = rewards[state] + (gamma * value)
			policy_matrix[state] = idx

		if np.allclose(temp, value_matrix, rtol=1e-07, atol=1e-07):
			print("Breaking iteration .. (" + str(ite) + ")")
			return

def post_convergence():
	global policy_matrix, value_matrix
	print("Post convergence...")

	for s in range(num_states):
		max_a = 0
		max_val = -99999
		for a in range(num_action):
			val = 0
			for s_t in range(num_states):
				if(a == 0):
					val = val + (prob_a[s][s_t] * value_matrix[s_t])
				elif(a == 1):
					val = val + (prob_b[s][s_t] * value_matrix[s_t])
				elif(a == 2):
					val = val + (prob_c[s][s_t] * value_matrix[s_t])
				elif(a == 3):
					val = val + (prob_d[s][s_t] * value_matrix[s_t])
			if(val > max_val):
				max_val = val
				max_a = a

		policy_matrix[s] = max_a

def direction(val):
	if val == 0:
		return "LEFT"
	elif val == 1:
		return "UP"
	elif val == 2:
		return "RIGHT"
	elif val == 3:
		return "DOWN"

def result_display(i):
	print("(" + str(i) + ") : " + direction(policy_matrix[i-1]))

def policy_iteration_mathod(ini):
	global policy_matrix, value_matrix
	print("Policy Iteration...")
	for i, val in enumerate(value_matrix):
		value_matrix[i] = 0

	for i, val in enumerate(policy_matrix):
		policy_matrix[i] = ini

	i = 0
	for ite in range(60):
		p = [[0 for x in range(num_states)] for y in range(num_states)]
		temp = copy.copy(value_matrix)
		for i in range(81):
			action = policy_matrix[i]
			if(action == 0):
				p[i] = prob_a[i]
			if (action == 1):
				p[i] = prob_b[i]
			if (action == 2):
				p[i] = prob_c[i]
			if (action == 3):
				p[i] = prob_d[i]
		p_matrix = np.matrix(p)

		value_matrix = np.dot(inv(np.identity(num_states) - gamma * p_matrix), np.matrix(rewards).transpose())

		for j in range(num_states):
			sum1 = 0.0
			for k in range(num_states):
				sum1 += prob_a[j][k] * value_matrix[k]
			sum2 = 0.0
			for k in range(num_states):
				sum2 += prob_b[j][k] * value_matrix[k]
			sum3 = 0.0
			for k in range(num_states):
				sum3 += prob_c[j][k] * value_matrix[k]
			sum4 = 0.0
			for k in range(num_states):
				sum4 += prob_d[j][k] * value_matrix[k]
			maxSum = max(sum1, sum2, sum3, sum4)
			if(maxSum == sum1):
				policy_matrix[j] = 0
			if (maxSum == sum2):
				policy_matrix[j] = 1
			if (maxSum == sum3):
				policy_matrix[j] = 2
			if (maxSum == sum4):
				policy_matrix[j] = 3

		if(np.allclose(temp, value_matrix)):
			print(ite, "Breaking Iteration...")
			break

	print("Value Function: ")
	print(np.around(value_matrix, decimals=6).flatten())
	print("\n")

	print("Optimal Policy Function: ")
	print(policy_matrix)
	print("\n")
	
	print("(Policy Iteration) -> Final Maze Output...")
	cells = [3, 11, 12, 15, 16, 17, 20, 22, 
			 23, 24, 26, 29, 30, 31, 34, 35,
			 39, 43, 48, 52, 53, 56, 57, 58,
			 59, 60, 61, 62, 66, 70, 71]

	for i in cells:
		result_display(i)

def value_iteration():
	global policy_matrix, value_matrix
	iteration_func()
	post_convergence()

	print("Value Function: ")
	print(np.around(value_matrix, decimals=6))
	print("\n")

	print("Optimal Policy Function: ")
	print(policy_matrix)
	print("\n")
	
	print("(Value Iteration) -> Final Maze Output...")
	cells = [3, 11, 12, 15, 16, 17, 20, 22, 
			 23, 24, 26, 29, 30, 31, 34, 35,
			 39, 43, 48, 52, 53, 56, 57, 58,
			 59, 60, 61, 62, 66, 70, 71]

	for i in cells:
		result_display(i)
	print("\n")

if __name__ == "__main__":
	read_data()
	value_iteration()
	policy_iteration_mathod(0) # for west at start
	policy_iteration_mathod(2) # for east at start
	policy_iteration_mathod(3) # for south at start
