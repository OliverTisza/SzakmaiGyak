import numpy as np
import pickle as pkl
import argparse
from time import time
import os
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
parser.add_argument('--NT', required=True,default=100, type=int)
args = parser.parse_args()
path = args.path
NT = args.NT

t = time()

with open(path+"\\cnet\\doc_top_matrix_NT"+str(NT)+".pkl","rb") as f:
		doc_top = pkl.load(f)

with open(path+"\\cnet\\top_term_matrix_NT"+str(NT)+".pkl","rb") as g:
		top_term = pkl.load(g)



doc_top = np.matrix(doc_top)
top_term = np.matrix(top_term)

doc_term = doc_top*top_term

doc_term = np.array(doc_term)

pkl.dump(doc_term,open(path+"\\cnet\\fontos_doc_term_matrix_NT"+str(NT)+".pkl","wb"))



print('Time for this WHOLE thing: {} mins'.format(round((time() - t) / 60, 2)))


