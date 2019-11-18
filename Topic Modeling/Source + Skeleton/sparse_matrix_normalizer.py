import gzip
import pickle as pkl
import scipy.sparse
import math
import sys
import argparse
from time import time
import numpy as np
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path


sparse_matrix = scipy.sparse.load_npz(path+'\\sparse_matrix.npz')
"""
szum_matrix = np.array(sparse_matrix.sum(axis=1))

sparse_matrix = sparse_matrix.toarray()

for i in range(len(sparse_matrix)):
    print(i)
    for j in range(len(sparse_matrix[0])):
        sparse_matrix[i][j] = sparse_matrix[i][j] / szum_matrix[i][0]
        if szum_matrix[i][0] == 0.0:
            sparse_matrix[i][j] = 0.0
"""            
            
normalized_matrix = preprocessing.normalize(sparse_matrix,norm ='l1',axis=1)
    
#normalized_matrix = scipy.sparse.csr_matrix(sparse_matrix)

scipy.sparse.save_npz(path+"\\normalized_sparse_matrix.npz",normalized_matrix)
