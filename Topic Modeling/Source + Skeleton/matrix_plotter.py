import numpy as np
import pickle as pkl
import argparse
from time import time
import os
import matplotlib.pyplot as plt


"""
Plotokat készít arra vonatkozóan, hogy egy dokumentumhoz az összes topic közül melyiket mennyire rendelte hozzá.
Ezek a plotok kerülnek cnet-en belül a cnet\\doc_top_plots mappába

A 33. sorban átírható hogy az első hány dokumentum plotját készítse el.

"""




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path

t = time()

with open(path+"\\cnet\\doc_top_matrix.pkl","rb") as f:
		doc_top = pkl.load(f)



plt.figure(figsize=(25,10))
for j in range(50): # <- hány dokumentum topic relevanciáját plotoljuk
	arr = []

	for i in range(len(doc_top[0])):
		if(doc_top[j][i] != 0):
			arr.append(i)
			

		
	
	plt.bar(np.arange(len(doc_top[j])),doc_top[j])
	plt.xticks(arr,arr)
	plt.ylabel('Relevance')
	
	plt.savefig(path+'\\cnet\\doc_top_plots\\doc_'+str(j)+'_topic_relevance.png')
	plt.clf()
   
