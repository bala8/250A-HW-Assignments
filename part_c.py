#Name: Balachander Padmanabha

import pprint
import math

# words from vocab
dict = {}
dict_idx = {}
dict_uni = {}

# individual word counts
word_count_mat = [[0 for x in range(500)] for y in range(500)]

def bigram(s, g):
  i = dict[s]
  j = dict[g]
  #print(g, s, float(word_count_mat[j][i])/dict_uni[j])
  return (float(word_count_mat[j][i])/dict_uni[j])

if __name__ == "__main__":
  log_uni = 1.0
  log_bi = 1.0
  denominator_uni = 0.0

  sentence = "Last week the stock market fell by one hundred points"
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
        log_uni = log_uni * (float(dict_uni[i]) / denominator_uni)


  with open("hw4_bigram.txt", "r") as ins:
    for i, line in enumerate(ins):
      line = line.rstrip("\n")
      i, j, count = line.split("\t")
      word_count_mat[int(i)-1][int(j)-1] = float(count) # as file is 1 indexed

  for x in range(0, len(words)):
    if x == 0:
      log_bi *= bigram(words[0], "<s>")
    else:
      log_bi *= bigram(words[x], words[x-1])

  print("Given sentence => " + sentence)
  print("Lu => " + str(math.log(log_uni)))
  print("Lb => " + str(math.log(log_bi)))

