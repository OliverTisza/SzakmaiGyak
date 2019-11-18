import numpy as np
import pickle as pkl
import argparse
from time import time
import os
from gensim import corpora, models
import gensim
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
parser.add_argument('--NT', required=True,default=100, type=int)
args = parser.parse_args()
path = args.path
NT = args.NT



with open(path+"\\LDA_NT_"+str(NT)+".pkl","rb") as f:
		model = pkl.load(f)


top_term = model.get_topics()


pkl.dump(top_term,open(path+"\\cnet\\top_term_matrix_NT"+str(NT)+".pkl","wb"))

matrix = np.matrix(top_term)


normalized_top_term_matrix = preprocessing.normalize(matrix,norm ='l1',axis=0)
pkl.dump(normalized_top_term_matrix,open(path+"\\cnet\\normalized_top_term_matrix_NT"+str(NT)+".pkl","wb"))

top_term = []


doc_top = [x[:] for x in [[0]*NT] * 1000]

i = 0

while True:
    try:
        with open(path+"\\cnet\\cnet_corpus.pkl","rb") as g:
            bow = pkl.load(g)

        topic_dist_pairs = model.get_document_topics(bow)

        for j in range(len(topic_dist_pairs)):
            doc_top[i][topic_dist_pairs[j][0]] = topic_dist_pairs[j][1]
        
        print(i)
        i += 1


    except:
        break



            
pkl.dump(doc_top,open(path+"\\cnet\\doc_top_matrix_NT"+str(NT)+".pkl","wb"))


