import numpy as np
import pickle as pkl
import argparse
from time import time
import os
from sklearn.preprocessing import normalize

"""
Egyelőre csak konzolra írat ki.

argmax-al kiválasztja a dokumentumhoz legjobban illő term-et.

Ezt megcsinálja az eredeti mátrixra illetve az l1,l2 norlmalizált változataira.

Egy halmazba pakolja a termek id-jét, végülis engem csak az érdekelne mennyi különböző van 1000 különböző dokumentumra.

"""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path

with open(path+"\\cnet\\fontos_doc_term_matrix.pkl","rb") as f:
		matrix = pkl.load(f)

with open(path+"\\cnet\\cnet_dict.pkl","rb") as g:
		dictionary = pkl.load(g)
"""
print(len(dictionary),"dict")
print(len(matrix),"sor(minden sor egy dokumentum)")
print(len(matrix[0]),"oszlop(minden oszlop egy concept=szó=term)")
"""
my_set = {0}

for i in range(len(matrix)):
	
	my_set.add(np.argmax(matrix[i]))

print(my_set)

"""------------"""

L1_matrix = normalize(matrix,norm='l1')

my_set = {0}

for i in range(len(L1_matrix)):
	my_set.add(np.argmax(L1_matrix[i]))

print(my_set)

"""------------"""

L2_matrix = normalize(matrix,norm='l2')

my_set = {0}

for i in range(len(L2_matrix)):
	my_set.add(np.argmax(L2_matrix[i]))


print(my_set)

