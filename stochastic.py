#Name: Balachander Padmanabha

import random
import math
import matplotlib.pyplot as plt
import numpy as np

# global list to plot the graph
datax = []
datay = []

def stochastic(sample_size, bit_set):
  bit  = []
  prob = 0.0
  alpha = 0.2
  bit_set_sum = 0
  total_sum = 0
  decimal_num = 0

  for i in range(0, sample_size):
    decimal_num = 0
    bit = []
    prob = 0.0
    temp = 0

    for j in range(0, 10):
      rand = random.uniform(0,1)
      if rand > 0.5:
        bit.append(1)
      else:
        bit.append(0)
      decimal_num = decimal_num + (math.pow(2,j)*bit[j])
    
    temp = abs(128 - decimal_num)
    prob = (1-alpha)/(1+alpha)*(math.pow(alpha,temp))
    
    # consider for full sample size
    total_sum = total_sum + prob

    # consider if relevant bit_set is set
    if bit[bit_set-1] == 1:
      bit_set_sum = bit_set_sum + prob
  
  print(sample_size, (bit_set_sum/total_sum))
  datax.append(sample_size)
  datay.append((bit_set_sum/total_sum))

if __name__ == "__main__":
  sample_size = 1000002
  # change value to set particular bit to 1
  bit_set = 2 
  
  # step size is 100000 for plotting the graph
  for i in range(1,sample_size,50000):
    stochastic(i, bit_set)

  # plotting the graph
  plt.plot(datax, datay, marker='o')
  var = "B-" + str(bit_set)
  plt.title(var)
  plt.ylabel('Probability')
  plt.xlabel('No of random samples')
  plt.xticks(np.arange(min(datax), max(datax)+1, 500000))
  plt.show()

