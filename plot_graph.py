# Name: Balachander Padmanabha

import matplotlib.pyplot as plt
import numpy as np
import math

def graph_gen(t):
	plt.title("6.3.c Plot for Q(x," + str(t) + ")")
	xn = 0
	data_x = []
	data_y = []
	
	for i in np.arange(-10, 10, 0.02):
		x = np.log(np.cosh(t)) + (np.tanh(t) * (i - t)) + ((1.0/2) * (i - t) * (i - t))
		data_x.append(i)
		data_y.append(x)
		
	lines = plt.plot(data_x, data_y)
  	plt.ylabel('Q(x,' + str(t) + ")")
  	plt.xlabel('Iteration (n)')
  	plt.setp(lines, color='r', linewidth=1.0)
  	plt.show()

def graph_gen_fx():
	plt.title("6.3.c Plot for f(x)=log(cosh(x))")
	xn = 0
	data_x = []
	data_y = []
	
	for i in np.arange(-10, 10, 0.02):
		x = np.log(np.cosh(i))
		data_x.append(i)
		data_y.append(x)
		
	lines = plt.plot(data_x, data_y)
  	plt.ylabel('f(x)')
  	plt.xlabel('Iteration (n)')
  	plt.setp(lines, color='r', linewidth=1.0)
  	plt.show()

def graph_gen_func():
	plt.title("6.3.h Plot for g(x)=(1/10) summation_of_k_1_to_10 [log(cosh(x+1/k))]")
	xn = 0
	data_x = []
	data_y = []
	
	for i in np.arange(-10, 10, 0.02):
		x = np.log(np.cosh(i))
		data_x.append(i)
		data_y.append(x)

	lines = plt.plot(data_x, data_y)
  	plt.ylabel('f(x)')
  	plt.xlabel('Iteration (n)')
  	plt.setp(lines, color='r', linewidth=1.0)
  	plt.show()

if __name__ == '__main__':
	#graph_gen_fx()
	#graph_gen(1)
	#graph_gen(-2)
	graph_gen_func()
	
	
