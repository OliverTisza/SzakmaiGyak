import pickle as pkl
import os
from scipy.sparse import csr_matrix
import scipy.sparse as sp

path = "SemCat\\Categories\\"
txts = os.listdir('SemCat\\Categories')

my_arr = []
i2w = dict()

for i in txts:
    f = open(path+i,'r')

    for line in f:
        line = line.replace("\n","")
        my_arr.append(line)
        

    
for i,v in enumerate(my_arr):
    i2w[i] = [v]


for i in txts:
    with open(path+i,'r') as f:
        texts = f.read()
        
    for k,v in i2w.items():
        
        if v[0] in texts:
            i2w[k].append(1)
        else:
            i2w[k].append(0)
    
sparse_matrix = []
final_i2w = dict()

for k,v in i2w.items():
    final_i2w[k] = v[0]
    sparse_matrix.append(v[1:])

pkl.dump(final_i2w,open("SemCat\\unprocessed_ws.pkl",'wb'))

matrix = csr_matrix(sparse_matrix)

sp.save_npz("SemCat\\sparse_matrix.npz",matrix)

print("semcat_prepare successful")

