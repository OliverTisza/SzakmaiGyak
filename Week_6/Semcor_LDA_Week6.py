import os
from time import time
import numpy as np
import gensim
import argparse
from gensim.models import CoherenceModel
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import pickle as pkl



def main():
	
	t = time()

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--path', required=False,default=".", type=str)
	args = parser.parse_args()
	path = args.path
	
	""" ------------------------------------------------- Bag of words -------------------------------------------------------- """


	with open(path+'\\documents.pkl','rb')as f:
		documents = pkl.load(f)	

	dictionary = gensim.corpora.Dictionary(documents)

	dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

	bow_corpus = [dictionary.doc2bow(doc) for doc in documents]


	""" ---------------------------------------- Coherence Values and Num Topics Graph ---------------------------------------- """

	def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):

		coherence_values = []
		model_list = []
		for num_topics in range(start, limit, step):
			print("Working on next model, num_topics =",num_topics,"...")
			model = gensim.models.LdaMulticore(corpus=bow_corpus, num_topics=num_topics, id2word=dictionary, passes=3, workers=3, random_state=0)
			model_list.append(model)
			coherencemodel = model.log_perplexity(bow_corpus)
			#coherencemodel = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_npmi')
			print("Perplexity: ",coherencemodel)
			coherence_values.append(coherencemodel)

		return model_list, coherence_values

	# Can take a long time to run.
	print("Computing coherence values...")
	model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=bow_corpus, texts=documents, start=2, limit=40, step=6)

	# Save graph
	limit=40; start=2; step=6;
	x = range(start, limit, step)
	plt.plot(x, coherence_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Perplexity score")
	plt.legend(("coherence_values"), loc='best')
	print('Time for this WHOLE thing: {} mins'.format(round((time() - t) / 60, 2)))
	plt.savefig(path+'\\perplex.png')

	

	""" --------------------------------------------------- LDA -------------------------------------------------------------- """

	
	print("\nWorking on simple LDA num_topics=7, passes=3...")


	lda_model_bow = gensim.models.LdaModel(bow_corpus, num_topics=7, id2word=dictionary, passes=3, random_state=0)

	f = open(path+"\\stats.txt",'w')
	
	for idx, topic in lda_model_bow.print_topics(-1):
		print('Topic: {} \nWords: {}'.format(idx, topic))
		f.write(str('\nTopic: {} \nWords: {}'.format(idx, topic)))
	

	print('\nPerplexity: ', lda_model_bow.log_perplexity(bow_corpus))
	f.write('\nPerplexity: '+str(lda_model_bow.log_perplexity(bow_corpus)))
	f.write('\n')

	"""
	coherence_model_lda = CoherenceModel(model=lda_model_bow, texts=documents, dictionary=dictionary, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)
	f.write('\nCoherence Score: '+str(coherence_lda))
	"""

	f.close()
	print('\nworking on topic visualization')
	
	vis = pyLDAvis.gensim.prepare(lda_model_bow, bow_corpus, dictionary)
	pyLDAvis.save_html(vis,path+'\\LDA_visualized.html')

	print('Time for this WHOLE thing: {} mins'.format(round((time() - t) / 60, 2)))

if __name__ == "__main__":
	main()
