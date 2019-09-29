import sys
from time import time
import numpy as np
import gensim
import argparse
from gensim.models import CoherenceModel
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
import pickle as pkl

"""
LDA modell és vizualizáció elkészítése, majd ezek kimentése.
num_topic átírása: 67.sor

A kimentett LDA neve : LDA_NT_<szám>.pkl ahol az NT a num_topics rövidítése lenne és utána a szám hogy mennyi volt a num_topics.

Futási idő(nekem):
	num_topic = 100, passes = 10 esetén : 137 perc míg elkészíti és kimenti a modellt + 40 perc míg elkészíti és kimenti a vizualizációt.
	num_topic = 250, passes = 10 esetén : 241 + 45 perc

	Régebbről kisebb num_topic-okon (7,14,20,25,50) 20-50 perc között
"""

def main():
	
	t = time()

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--path', required=False,default=".", type=str)
	args = parser.parse_args()
	path = args.path
	
	""" -------------------------------------------------
		Init
	    -------------------------------------------------
	"""


	with open(path+"\\cnet\\cnet_dict.pkl","rb") as g:
		dictionary = pkl.load(g)
			

	h = open(path+"\\cnet\\cnet_corpus.pkl","rb")

	bow_corpus = []

	"""

	A pickle soronként olvas, jelen esetben egy sorban egy dokumentum szózsákja van.
	
	"""

	while True:
			try:
				bow_corpus.append(pkl.load(h))
				
			except:
				break

	
	h.close()

	
	""" ------------------------------------------------- LDA and PyLDAvis ---------------------------------------------------- """

	num_topic = 100

	print("Working on model, num_topics =",num_topic,"...")
	
	model = gensim.models.LdaMulticore(corpus=bow_corpus, num_topics=num_topic, id2word=dictionary, passes=10, workers=3, random_state=0)
	pkl.dump(model,open(path+"\\LDA_NT_"+str(num_topic)+".pkl","wb"))
	print('Time for model: {} mins'.format(round((time() - t) / 60, 2)))

	
	print('\nworking on topic visualization')
	
	vis = pyLDAvis.gensim.prepare(model, bow_corpus, dictionary)
	pyLDAvis.save_html(vis,path+'\\LDA_visualized.html')
	
	
	print('Time for vis: {} mins'.format(round((time() - t) / 60, 2)))
if __name__ == "__main__":
	main()
