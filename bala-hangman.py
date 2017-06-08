# Student Name: Balachander Padmanabha
# Email id: bpadmana@ucsd.edu

#!/usr/bin/python
import pprint
import heapq
import re

# dictionary to store the words read from file
dict = {}

# array for characters holding all the words
word_list = []

# variable to store temp results
P_of_E_given_W = []
P_of_w = []
sum_of_posterior_deno = 0.0

# total word count variables
words_count = 0

# open file to read the word frequencies
def read_inputs():
  with open("hw1_word_counts_05.txt", "r") as ins:
      global words_count
      for line in ins:
          line = line.rstrip("\n")
          word, count = line.split(" ")
          dict[word] = int(count)
          word_list.append((word))
          words_count = words_count + int(count)

def mostfrequent(n):
    # get 'n' most occuring words in dict
    print(str(n) + " most frequent words:")
    print(heapq.nlargest(n, dict, key=dict.get))

def leastfrequent(n):
    # get 'n' least occuring words in dict
    print(str(n) + " least frequent words:")
    print(heapq.nsmallest(n, dict, key=dict.get))

def prob_util_1(correct, incorrect):
    i = 0 
    for word in word_list:
        check = 0

        for c in incorrect:
            if(word.find(c) != -1): # c found in word
                P_of_E_given_W.append(0)
                check = 1
                break
        
        if(check == 1):
            check = 0
            i = i + 1
            continue

        for idx, c in enumerate(correct):
            if(c is None):
                # skip the non-guessed positions
                continue
                
            if(word.find(c) != -1): # c found in word
                if(word[idx] != c):
                    # c not found at correct position of word in consideration
                    P_of_E_given_W.append(0)
                    check = 1
                    break
                # checking for cases with correct guess character repeating 
                # multiple times in the word in consideration
                temp = 0
                for m in re.finditer(c, word):
                    if(temp > 1):
                      break
                    temp = temp + 1
                if(temp > 1):
                    P_of_E_given_W.append(0)
                    check = 1
                    break
            else:
                # c not found in word under consideration
                P_of_E_given_W.append(0)
                check = 1
                break

        if(check == 1):
            check = 0
            i = i + 1
            continue
        
        P_of_E_given_W.append(1)
        i = i + 1

def prob_util_2():
    count_w = 0.0

    for i, word in enumerate(word_list):
        count_w = dict[word]
        P_of_w.append(count_w/float(words_count))

def summation_util():
    global sum_of_posterior_deno
    
    t = 0
    for i, word in enumerate(word_list):
        t = P_of_E_given_W[i] * P_of_w[i]
        sum_of_posterior_deno = sum_of_posterior_deno + t

def next_guess(num, correct, incorrect):
    print("Next Guess for Test case: " + str(num))
    print(correct)
    print(incorrect)
    prob_max = 0.0
    c_final = "#"
    p = 0

    # calculate P(E|W=w) for all words
    prob_util_1(correct, incorrect)

    #calculate P(W=w) for all words
    prob_util_2()

    # calculate denominator of posterior probability function
    summation_util()
    
    # calculate Probability for all the letters 
    for i in range(26):
      c = chr(i+65)
      
      if(c in incorrect):
          #no point calculating for incorrect charaters
          continue

      if(c in correct):
          #no point calculating for correct characters also
          continue

      res_c = 0.0
      #calculate P(Li=l for some i in {1,2,3,4,5} | W=w) for all words
      for i, word in enumerate(word_list):
          if(word.find(c) != -1): # c found in word
              p = 1.0
          else:
              p = 0.0

          res_c = res_c + (p * ((P_of_E_given_W[i] * P_of_w[i])/sum_of_posterior_deno))

      if(res_c > prob_max):
          prob_max = res_c
          c_final = c
          res_c, c = 0.0, "#"

    print("Best next guess l => " + str(c_final))
    print("P(Li=l for some i in {1,2,3,4,5} | E) => " + str(round(prob_max, 4)))

if __name__ == "__main__":
    read_inputs()

    # sanity check functions
    mostfrequent(15)
    print("\n")
    leastfrequent(14)
    print("\n")
   
    # IMP: each test case needs to run seperately.

    correct = [None, None, None, None, None]
    incorrect = []
    #next_guess(1, correct, incorrect)
    print("\n")
    
    correct = [None, None, None, None, None]
    incorrect = ['A','I']
    #next_guess(2, correct, incorrect)
    print("\n")
    
    correct = ['A', None, None, None, 'R']
    incorrect = []
    #next_guess(3, correct,incorrect)
    print("\n")
    
    correct = ['A', None, None, None, 'R']
    incorrect = ['E']
    #next_guess(4, correct,incorrect)
    print("\n")
    
    correct = [None, None, 'U', None, None]
    incorrect = ['O', 'D', 'L', 'C']
    #next_guess(5, correct,incorrect)
    print("\n")
    
    correct = [None, None, None, None, None]
    incorrect = ['E', 'O']
    #next_guess(6, correct,incorrect)
    print("\n")
    
    correct = ['D', None, None, 'I', None]
    incorrect = []
    #next_guess(7, correct,incorrect)
    print("\n")
    
    correct = ['D', None, None, 'I', None]
    incorrect = ['A']
    #next_guess(8, correct,incorrect)
    print("\n")
    
    correct = [None, 'U', None, None, None]
    incorrect = ['A', 'E', 'I', 'O', 'S']
    next_guess(9, correct,incorrect)
    print("\n")

