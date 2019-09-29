from time import time
import argparse
import pickle as pkl
import gensim

"""
Gensim-es dictionary-t készít és ment el

Egy dictionary egy dokumentumból van, aztán az eredeti dictionary-hez merge-el egy újat amíg van új.
Ha már megvan az egy nagy dictionary-nk filterezzük, és csak 100k szót tartunk meg.

A dokumentáció szerint a merge_with() metódus:
	Merge another dictionary into this dictionary, mapping the same tokens to the same ids and new tokens to new ids.

--------------------

Return : cnet_dict.pkl a cnet-es mappában
--------------------

A cnet_dict és az unprocessed_ws elméletben ugyanazt tartalmazza, de más típusú objektumok, és az egyiknek van doc2bow metódusa,
a másiknak pedig nincs
--------------------

TODO: Lehet ez a lépés átugorható lenne ha írnék a sima dict-nek egy doc2bow metódust.

"""

t = time()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path

	
"""

Egyszerre csak annyi dokumentumot tölt be amennyi belefér a memóriába (jelen esetben 50db)

"""
			
	
for j in range(0,1000,50):
				
	documents = []             
	for i in range(j,j+50):
		with open(path+'\\cnet\\cnet_docs\\cnet_document'+str(i)+'.pkl','rb')as file:
			documents.append(pkl.load(file))
		
	if j == 0:
		dictionary = gensim.corpora.Dictionary(documents)
		
	else:
		dictionary2 = gensim.corpora.Dictionary(documents)
		transformer = dictionary.merge_with(dictionary2)
		
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)	
pkl.dump(dictionary,open(path+"\\cnet\\cnet_dict.pkl","wb"))
print('Time for creating the dict: {} mins'.format(round((time() - t) / 60, 2)))
	
