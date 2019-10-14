import gzip
import pickle as pkl
import scipy.sparse
import math
import sys
import argparse
from time import time
import numpy as np


"""

Kimenti egy txt-be az összes dokumentum top tíz beágyazott szavát (nem a conceptet)

Elméletileg ezzel a 10 szóval kellene összehasonlítanom az LDA egy hozzárendelt termjét

"""





parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path

t = time()

h = open(path+"\\cnet\\cnet_corpus.pkl","rb")

with open(path+"\\cnet\\cnet_dict.pkl","rb") as g:
		dictionary = pkl.load(g)

with open(path+"\\unprocessed_ws.pkl","rb") as h:
		i2w = pkl.load(h)

top10 = open(path+"\\top10words_per_doc.txt",'w',encoding='utf8')

sparse_matrix = scipy.sparse.load_npz(path+'\\sparse_matrix.npz')


for i in range(1000):

	try:
	
		top10.write("DOC: "+str(i)+"\n")
		top10.write("--------------------------\n")
		column = sparse_matrix.getcol(i)
		
		for j in range(10):            
			top10.write(i2w[column.argmax()]+"\n")
			column[column.argmax()] = 0
			
		top10.write("--------------------------\n")

	except:
		break


top10.close()


print('Time for top10: {} mins'.format(round((time() - t) / 60, 2)))

