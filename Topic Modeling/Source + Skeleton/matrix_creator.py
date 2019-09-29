import numpy as np
import pickle as pkl
import argparse
from time import time
import os
from gensim import corpora, models
import gensim

"""
Dokumentum-topic és topic-term mátirx létrehozása.

model.get_topic_topics(bow) :

returns list of (int, float) – Topic distribution for the whole document.
Each element in the list is a pair of a topic’s id, and the probability that was assigned to it.
--------------------------------------
Emlékeztető:

A cnet_corpus-ban egy sor egy dokumentum szózsákja, így soronként feldolgozható.
--------------------------------------

Át kell írni, hogy melyik modellt töltse be (38.sor)
Ha nem 100-as a topicszám azt az 50. sorban kell átírni


"""



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path




with open(path+"\\LDA_NT_100.pkl","rb") as f:
		model = pkl.load(f)


top_term = model.get_topics()


pkl.dump(top_term,open(path+"\\cnet\\top_term_matrix.pkl","wb"))

top_term = [] #kiürítem hogy ne foglaljon helyet


doc_top = [x[:] for x in [[0]*100] * 1000]   #100 a topicszám, 1000 a dokumentumszám

i = 0

while True:
	try:
		with open(path+"\\cnet\\cnet_corpus.pkl","rb") as g:
			bow = pkl.load(g)
			
		topic_dist_pairs = model.get_document_topics(bow)

		for j in range(len(topic_dist_pairs)):
			doc_top[i][topic_dist_pairs[j][0]] = topic_dist_pairs[j][1]

		i += 1

	except:
		break

"""
Megpróbáltam a legjelentősebbet kinullázni mert mindegyiknél ugyanaz volt, eredménykent egy helyett 3-5 topic jelent meg
"""
#for i in range(len(doc_top)):
#    doc_top[i][np.argmax(doc_top[i])] = 0

			
pkl.dump(doc_top,open(path+"\\cnet\\doc_top_matrix.pkl","wb"))


