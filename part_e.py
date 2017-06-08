#Name: Balachander Padmanabha

import pprint
import math
import numpy as np
import matplotlib.pyplot as plt

# words from vocab
dict = {}
dict_idx = {}
dict_uni = {}

# individual word counts
word_count_mat = [[0 for x in range(500)] for y in range(500)]

def bigram(s, g):
  i = dict[s]
  j = dict[g]
  if (float(word_count_mat[j][i])/dict_uni[j]) == 0.0:
    #print("not possible => (" + g + ") (" + s + ") as prob is 0.0")
    return 0 
  else:
    return(float(word_count_mat[j][i])/dict_uni[j])

if __name__ == "__main__":
  log_uni = 1.0
  log_bi = 1.0
  denominator_uni = 0.0
  final = []
  pm = []

  sentence = "The nineteen officials sold fire insurance"
  sentence = sentence.upper()
  words = sentence.split(" ")

  with open("hw4_vocab.txt", "r") as vocab:
    for i, line in enumerate(vocab):
      line = line.rstrip("\n")
      dict[line] = i
      dict_idx[i] = line
  
  with open("hw4_unigram.txt", "r") as ins:
    for i, line in enumerate(ins):
      line = line.rstrip("\n")
      dict_uni[i] = int(line)
      denominator_uni += float(line)

  with open("hw4_unigram.txt", "r") as ins:
    for i, line in enumerate(ins):
      if dict_idx[i] in words:
        temp = (dict_idx[i], float(dict_uni[i]) / denominator_uni)
        final.append(temp)

  with open("hw4_bigram.txt", "r") as ins:
    for i, line in enumerate(ins):
      line = line.rstrip("\n")
      i, j, count = line.split("\t")
      word_count_mat[int(i)-1][int(j)-1] = float(count) # as file is 1 indexed

  for x in range(0, len(words)):
    if x == 0:
      for y in final:
        if y[0] == words[x]:
          log_uni = y[1]
      log_bi = bigram(words[0], "<s>")
      temp = (words[0], "<s>", log_uni, log_bi)
      pm.append(temp)
    else:
      for y in final:
        if y[0] == words[x]:
          log_uni = y[1]
      log_bi = bigram(words[x], words[x-1])
      temp = (words[x], words[x-1], log_uni, log_bi)
      pm.append(temp)
  
  datax = []
  datay = []
  point = 1
  maxi = -10000
  max_l = 0

  for lambda_val in np.arange(0,1, 0.008):
    for x in range(0, len(words)):
      if x == 0:
        for p in pm:
          if p[0] == words[0] and p[1] == "<s>":
            log_uni = p[2]
            log_bi = p[3]
      else:
        for p in pm:
          if p[0] == words[x] and p[1] == words[x-1]:
            log_uni = p[2]
            log_bi = p[3]
      point *= (((1 - lambda_val)*log_uni) + ((lambda_val)*log_bi))
    
    if math.log(point) > maxi:
      maxi = math.log(point);
      max_l = lambda_val
      
    datax.append(math.log(point))
    datay.append(lambda_val)
    log_uni = log_bi = point = 1
  
  plt.plot(datay, datax, marker='o')
  print("MAX LAMBDA value => " + str(max_l))
  var = "Log-Likelihood Lm Plot"
  plt.title(var)
  plt.ylabel('Lm')
  plt.xlabel('Lambda')
  plt.show()

