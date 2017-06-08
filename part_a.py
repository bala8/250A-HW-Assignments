#Name: Balachander Padmanabha

import re

# words from vocab
dict = {}

# individual word counts
word_count_a = {}

if __name__ == "__main__":
    denominator = 0.0
    with open("hw4_vocab.txt", "r") as vocab:
		    for i, line in enumerate(vocab):
			      line = line.rstrip("\n")
			      dict[i] = line
  
    with open("hw4_unigram.txt", "r") as ins:
        for i, line in enumerate(ins):
            line = line.rstrip("\n")
            if re.match("^A", dict[i]) is not None:
                word_count_a[dict[i]] = int(line)
            denominator += int(line)
    i = 1
    for key, value in word_count_a.items():
        print(str(i) + ". (" + key + ") prob: " + str(value/denominator))
        i += 1
