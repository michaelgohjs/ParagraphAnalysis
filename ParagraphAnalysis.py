#import nltk
#nltk.download('punkt')

from nltk.tokenize import LineTokenizer
from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import string

from collections import Counter, deque
from nltk.tokenize import regexp_tokenize
import pandas as pd

#Input Selection
fileName = "paragraph.txt"

#Import data
fileObj = open(fileName, "r") #opens the file in read mode
Lines = fileObj.read().splitlines() #puts the file into list
fileObj.close()

fileObj = open(fileName, "r") #opens the file in read mode
words = fileObj.read() #puts the file into string
fileObj.close()


#Probability of word "data" in each of the 22 lines
Prob =[]
x =[]
count=0
for line in Lines:
    WordToken=word_tokenize(line) 
    fdist = FreqDist(WordToken)
    Prob.append(fdist.freq('data'))
    x.append(count+1)
    count+=1    

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.set_title("Probability of 'Data' occurrence")
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xlabel('Line number')
ax.bar(x,Prob)
plt.show()

#Distribution of distinct words across all 22 lines
words=words.translate(str.maketrans('','',string.punctuation))#Remove punctuation
WordToken=word_tokenize(words)
fdist = FreqDist(WordToken)
fdist.plot(len(fdist),cumulative=False)
plt.show()
#Do not show words that only occur once
fdist.plot(len(fdist)-len(fdist.hapaxes()),cumulative=False)
plt.show()


#Probability of the word "analytics" occurring after the word "data"
def grouper(iterable, length=2):
    i = iter(iterable)
    q = deque(map(next, [i] * length))
    while True:
        yield tuple(q)
        try:
            q.append(next(i))
            q.popleft()
        except StopIteration:
            break

def tokenize(text):
    return [word.lower() for word in regexp_tokenize(text, r'\w+')]

def follow_probability(word1, word2, vec):
    subvec = vec.loc[word1]
    print(subvec)
    try:
        ct = subvec.loc[word2]
        print(ct)
    except:
        ct = 0
    return float(ct) / (subvec.sum() or 1)


tokens = tokenize(words)
markov = list(grouper(tokens))
vec = pd.Series(Counter(markov))
Probability = follow_probability('data', 'analytics', vec)
print("Probability of 'Analytics' occuring after 'Data' is:")
print(Probability)














