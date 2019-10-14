import gzip
import pickle as pkl
import scipy.sparse
import math
import sys
import argparse
from time import time
import numpy as np

"""

Elkészíti a dokumentumokat és kimenti azokat. (dokumentum = dimenzió)
Egy fájl egy dokumentum.

Paraméterek :

	Path : Ahonnan betölti a szükséges fájlokat (i2w ,concepts, ritka mátrix)

--------------

A dokumentumokat a Path+\\cnet\\cnet_docs mappába próbálja menteni (erre adtam neked egy vázat remélem)

--------------

Return : 1000 fájl cnet_docs mappában

--------------

Név konvenció : cnet_document<sorszám>.pkl

pl.:  cnet_document0.pkl   (első dokumentum)
      cnet_document999.pkl (utolsó dokumentum)
"""


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path

t = time()

with open(path+'\\unprocessed_ws.pkl','rb') as f:
	i2w = pkl.load(f)

with open(path+'\\concepts.pkl','rb') as g:
	concepts = pkl.load(g)


sparse_matrix = scipy.sparse.load_npz(path+'\\sparse_matrix.npz')

arr_matrix = sparse_matrix.toarray()

documents = [x[:] for x in [[]] * len(arr_matrix[0])]

"""

Először feltölti az összes dokumentumot, aztán kimenti az osszes dokumentumot.

Minden iterációban egy szót(termet) minden előfordulásához beírjuk, ezért mind az 50k iteráción végig kell menjünk, mert csak
akkor lesz biztosan minden dokumentum 100%-osan feltöltve.
--------------

TODO: Megnézni egyszerűbb/gyorsabb/biztonságosabb-e ha transzponálom a ritka mátrixot, akkor minden iterációban ki tudnék menteni
      egy dokumentumot gondolom


"""

for j in range (0,50000):#len(arr_matrix)... általánosságban len(arr_matrix) = len(i2w)-vel de nekem nem lásd.: cnet.py
	tmp_nonzeroes = []
	docs = []
	word = j

	try:
	
		for i in range(0,len(arr_matrix[j])):
			if arr_matrix[j][i] != 0.0:
				
				#print(str(j)+".-edik szó",str(i)+".-edik dokumentumban: ",arr_matrix[j][i])
				tmp_nonzeroes.append(arr_matrix[j][i])
				docs.append(i)

		
		
		for i,num in enumerate(tmp_nonzeroes):
			#for k in range(0,math.ceil((num/sum(tmp_nonzeroes))*len(arr_matrix[0]))):  <- Vanda hackje
			for l in range(len(concepts[i2w[j]])):
				documents[docs[i]].append(concepts[i2w[j]][l])
	except:
		break

	
				
print('Make docs time: {} mins'.format(round((time() - t) / 60, 2)))

# Tovább tart mint gondoltam, don't panic
for i in range(len(documents)):
	pkl.dump(documents[i],open(path+"\\cnet\\cnet_docs\\cnet_document"+str(i)+".pkl","wb"))
print('Make docs save time: {} mins'.format(round((time() - t) / 60, 2)))

