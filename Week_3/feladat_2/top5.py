from gensim.models import Word2Vec
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

f = open("szavak.txt",'r')
#egy sorban van az osszes szo
w = f.readline()

def tops(words):

    DIMENSION = 5
    
    words = words.split()

    model = Word2Vec([words], min_count = 1,size=DIMENSION, window = 3)

    eredmeny = [0]*DIMENSION

    for i in range(0,5):
        eredmeny += model.wv[words][i]
  
    eredmeny = np.array(eredmeny) 
    eredmeny = np.divide(eredmeny, len(words))

    top = [0]*len(words)
    
    for i in range(0,len(words)):
        top[i] = cosine_similarity(eredmeny.reshape(1,-1),model.wv[words[i]].reshape(1,-1))

    dictx = {}


    for i in range(0,len(top)):
        dictx[words[i]] = top[i]

    topfive = sorted(dictx, key=dictx.__getitem__)
    topfive = topfive[:len(topfive)-6:-1]

    return topfive

print("\nAtlagvektorhoz legkozelebbi 5 szo: \n",tops(w))
