import numpy as np
import pickle as pkl
import argparse
from time import time
import os
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pylab as plt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
parser.add_argument('--NT', required=True,default=100, type=int)
args = parser.parse_args()
path = args.path
NT = args.NT

with open(path+"\\cnet\\fontos_doc_term_matrix_NT"+str(NT)+".pkl","rb") as f:
		matrix = pkl.load(f)

with open(path+"\\cnet\\cnet_dict.pkl","rb") as g:
		dictionary = pkl.load(g)



"""------------"""

L1_matrix = normalize(matrix,norm='l1',axis=1)


"""
plt.figure(figsize=(25,10))
ax = sns.heatmap(L1_matrix[0:900])
plt.savefig(path+'\\cnet\\doc_term_plots\\heatmap_doc_0_900_NT'+str(NT)+'_col_norm1.png')
plt.clf()

"""
plt.figure(figsize=(25,10))
ax = sns.heatmap(L1_matrix[0:900])
plt.savefig(path+'\\cnet\\doc_term_plots\\heatmap_doc_0_900_NT'+str(NT)+'_row_norm1.png')
plt.clf()


