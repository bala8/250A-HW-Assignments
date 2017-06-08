#Name: Balachander Padmanabha

import re
import pprint
from operator import itemgetter

# words from vocab
dict = {}

# individual word counts
word_count_mat = [[0 for x in range(501)] for y in range(501)]

if __name__ == "__main__":
  the_index = 0
  denominator = 0.0
  final = []

  with open("hw4_vocab.txt", "r") as vocab:
    for i, line in enumerate(vocab):
      line = line.rstrip("\n")
      dict[i+1] = line
      if line == "THE":
        the_index = i+1
  
  with open("hw4_bigram.txt", "r") as ins:
    for i, line in enumerate(ins):
      line = line.rstrip("\n")
      i, j, count = line.split("\t")
      if int(i) == the_index:
        denominator += int(count)
        temp = (dict[int(j)], int(count))
        final.append(temp)
      word_count_mat[int(i)][int(j)] = int(count)
   
  final.sort(key=itemgetter(1), reverse=True)
  print("\nFive most likely words to follow THE are,")
  print("Sl.No. Word => Probability")
  for idx, val in enumerate(final):
    if idx < 5:
      print(str(idx) + ". " + str(val[0]) + " => " + str(val[1]/denominator))
  print("")
