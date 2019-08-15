import gzip
import pickle as pkl
import scipy.sparse
import math
import sys
import argparse
from time import time
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path


#print("len arr_matrix row : ",len(arr_matrix))
#print("\nlen arr_matrix col : ",len(arr_matrix[0]))
#print("arr_matrix[0]: ",arr_matrix[0][155],"\n")    # 0.szó (i2w.pkl) 155.dokumentum
#sor = szó (ugyanaz a szó más dokumentumokban)
#oszlop = dokumentum (szavak/tokenek listája)

t = time()

with open(path+'\\i2w.pkl','rb') as f:
	i2w = pkl.load(f)


sparse_matrix = scipy.sparse.load_npz(path+'\\sparse_matrix.npz')

arr_matrix = sparse_matrix.toarray()


#bow_corpus = [x[:] for x in [[]] * len(arr_matrix[0])]

documents = [x[:] for x in [[]] * len(arr_matrix[0])]



for j in range (0,len(arr_matrix)):
	tmp_nonzeroes = []
	docs = []
	word = j
	
	for i in range(0,len(arr_matrix[j])):
		if arr_matrix[j][i] != 0.0:
			
			print(str(j)+".-edik szó",str(i)+".-edik dokumentumban: ",arr_matrix[j][i])
			tmp_nonzeroes.append(arr_matrix[j][i])
			docs.append(i)

			
	for i,num in enumerate(tmp_nonzeroes):
		#print(docs[i],"dokumentumban: ",math.ceil((num/sum(tmp_nonzeroes))*len(arr_matrix[0])))
		#pair = (word,math.ceil((num/sum(tmp_nonzeroes))*len(arr_matrix[0])))
		#bow_corpus[docs[i]].append(pair)
		for k in range(0,math.ceil((num/sum(tmp_nonzeroes))*len(arr_matrix[0]))):
                        documents[docs[i]].append(i2w[j])
	
				
print('Time for this WHOLE thing: {} mins'.format(round((time() - t) / 60, 2)))


#pkl.dump(bow_corpus,open(path+"\\corpus.pkl","wb"))
pkl.dump(documents,open(path+"\\documents.pkl","wb"))
print('Time for this WHOLE thing: {} mins'.format(round((time() - t) / 60, 2)))

