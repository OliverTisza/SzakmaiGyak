import numpy as np
import pickle as pkl
import argparse
from time import time
import os
import matplotlib.pyplot as plt

"""

betölt és összeszoroz két mátrixot majd a szorzatukat kimenti tömbként

"""




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path

t = time()

with open(path+"\\cnet\\doc_top_matrix.pkl","rb") as f:
		doc_top = pkl.load(f)

with open(path+"\\cnet\\top_term_matrix.pkl","rb") as g:
		top_term = pkl.load(g)



doc_top = np.matrix(doc_top)
top_term = np.matrix(top_term)

doc_term = doc_top*top_term

doc_term = np.array(doc_term)

pkl.dump(doc_term,open(path+"\\cnet\\fontos_doc_term_matrix.pkl","wb"))

print('Time for this WHOLE thing: {} mins'.format(round((time() - t) / 60, 2)))


