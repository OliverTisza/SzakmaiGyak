import gzip
import argparse
import os
import numpy as np
import utils
import scipy.sparse.linalg
import scipy.sparse as sp
import pickle
import sys
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from time import time


"""
uj paraméter : savepath, ahova elmenti az i2w-ket és a ritka mátrixot
raw_i2w : nincs preprocesszelve (a cnet-hez kell)

"""

class SparseEmbedding(object):
	
	def __init__(self, embedding_path, savepath, languages=None, filter_rows=-1):
		self.i2w, self.W =  self.load_embeddings(embedding_path, savepath, languages)

	def load_embeddings(self, path, savepath, languages=None,  filter_rows=-1):
		"""
		Reads in the sparse embedding file.
		Parameters
		----------
		path : location of the gzipped sparse embedding file
		languages : a set containing the languages to filter for. 
		If None, no filtering takes plce.
		filter_rows : indicates the number of lines to read in.
		If negative, the entire file gets processed.
	
		
		
		Returns
		-------
		w2i : wordform to identifier dictionary
		i2w : identifier to wordform dictionary
		W : the sparse embedding matrix
		"""

		t = time()

		if type(languages) == str:
			languages = set([languages])
		elif type(languages) == list:
			languages = set(languages)

		i2w = dict()
		raw_i2w = dict()
		data, indices, indptr = [], [], [0]
		with gzip.open(path, 'rb') as f:
			for line_number, line in enumerate(f):
				if line_number == filter_rows:
					break
				parts = line.decode("utf-8").strip().split()
				language = parts[0][0:2]
				
				if languages is not None and language not in languages:
					continue


				if parts[0] not in STOPWORDS and len(parts[0]) > 3:
					i2w[len(i2w)] = PorterStemmer().stem(WordNetLemmatizer().lemmatize(parts[0], pos='v'))
					raw_i2w[len(raw_i2w)] = parts[0]
					
					for i, value in enumerate(parts[1:]):
						"""
						Előfordult, hogy nem csak a 0. hanem az 1. indexen is szó volt, ezeket átugrottam
						"""
						try:
							value = float(value)
							
						except ValueError:
							i2w.pop(len(i2w))
							raw_i2w.pop(len(raw_i2w))
							break
						
						if value != 0:
							data.append(float(value))
							indices.append(i)
							
					indptr.append(len(indices))
		
		W = sp.csr_matrix((data, indices, indptr), shape=(len(indptr)-1, i+1))
		

		sp.save_npz(savepath+"\\sparse_matrix.npz",W)
		pickle.dump(raw_i2w,open(savepath+"\\unprocessed_ws.pkl","wb"))
		pickle.dump(i2w,open(savepath+"\\i2w.pkl","wb"))
		

		print('Sparse embedding time: {} mins'.format(round((time() - t) / 60, 2)))		
		
		return i2w, W
	       


def main():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--embedding-location', required=True, type=str)
	parser.add_argument('--language', required=False, default=None, nargs='*', type=str)
	parser.add_argument('--savepath', required=False, default=".", type=str)
	args = parser.parse_args()
	print("The command line arguments were ", args)
	print(type(args.savepath))
	se = SparseEmbedding(args.embedding_location, args.savepath, args.language)

	print('{} words read in...'.format(se.W.shape[0]))

if __name__ == "__main__":
	main()
