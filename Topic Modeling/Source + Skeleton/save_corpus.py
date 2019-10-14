from time import time
import argparse
import pickle as pkl
import gensim


"""

Szinte ugyanaz mint a save_dict.py

A pickle-nek használom ki egy tulajdonságát, hogy egyszerre egy sort olvas be és egy sort ír.
A fájlt cnet_corpus.pkl néven menti.
--------------------

A cnet_corpus.pkl-ben egy sor egy dokumentum szózsákja.
Ezt a dokumentumok beolvasásának limitációja miatt csináltam így.

--------------------
Fontos:

a cnet_corpus.pkl fájlhoz hozzáfűzök, úgyhogy ha valami miatt többször futtatod ezt a kódot fontos hogy ne legyen vagy üres legyen
ezen fájlod, mert nem fogja felülírni.


"""


t = time()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path

""" --------------------------------------------------------------------------------------------------------- """



with open(path+"\\cnet\\cnet_dict.pkl","rb") as g:
		dictionary = pkl.load(g)

		
				

for j in range(0,1000,50):
	try:
		documents = []

		for i in range(j,j+50):
			with open(path+'\\cnet\\cnet_docs\\cnet_document'+str(i)+'.pkl','rb')as file:
				documents.append(pkl.load(file))
					
					
		bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
			
		for i in range(len(bow_corpus)):
				pkl.dump(bow_corpus[i],open(path+"\\cnet\\cnet_corpus.pkl","ab"))

	except:
		break
		
print('Time for creating the corpus: {} mins'.format(round((time() - t) / 60, 2)))
	
