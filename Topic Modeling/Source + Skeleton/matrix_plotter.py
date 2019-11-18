import numpy as np
import pickle as pkl
import argparse
from time import time
import os
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
parser.add_argument('--NT', required=True,default=100, type=int)
args = parser.parse_args()
path = args.path
NT = args.NT

t = time()

with open(path+"\\cnet\\doc_top_matrix_NT"+str(NT)+".pkl","rb") as f:
		doc_top = pkl.load(f)

with open(path+"\\cnet\\normalized_top_term_matrix_NT"+str(NT)+".pkl","rb") as g:
		top_term = pkl.load(g)
		

plt.figure(figsize=(35,15))
ax = sns.heatmap(doc_top)
plt.savefig(path+'\\cnet\\doc_top_plots\\document_topic_relevance_heatmap_NT'+str(NT)+'.png')
plt.clf()

plt.figure(figsize=(35,15))
ax = sns.heatmap(top_term)
plt.savefig(path+'\\cnet\\top_term_plots\\normalized_topic_term_relevance_heatmap_NT'+str(NT)+'.png')
plt.clf()

