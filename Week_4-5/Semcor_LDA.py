import os
from time import time
import xml.etree.ElementTree as ET
import numpy as np
import nltk
import gensim
from gensim.models import CoherenceModel
from gensim import corpora, models

#kesbetusit es kiveszi az irasjeleket
from gensim.utils import simple_preprocess

#specialis szavak amelyek elhagyhatoak
from gensim.parsing.preprocessing import STOPWORDS

#jelenidobe konvertal, szemelyes nevmasokat konvertal
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt



def main():
    
    t = time()
    tree = ET.parse('WSD_Training_Corpora\SemCor\semcor.data.xml')
    root = tree.getroot()
    
    """
    root          - corpus
    root[0]       - text (document)
    root[0][0]    - sentence 
    root[0][0][0] - word (token)

    """

    documents = [0]*len(root)
    i = 0



    for text in root:
        document = ''
        result = []
        for sentence in text:
            for word in sentence:
                document = document + " " + word.text
                    
        #egy dokumentum
        document = simple_preprocess(document)
        for token in document:
            if token not in STOPWORDS and len(token) > 3:
                result.append(PorterStemmer().stem(WordNetLemmatizer().lemmatize(token, pos='v')))

        documents[i] = result
        print(i)
        i += 1  


    """ ------------------------------------------------- Bag of words -------------------------------------------------------- """

    dictionary = gensim.corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]




    """ ---------------------------------------- Coherence Values and Num Topics Graph ---------------------------------------- """


    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):

        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            print("Working on next model, num_topics =",num_topics,"...")
            model = gensim.models.LdaMulticore(corpus=bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=3)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    # Can take a long time to run.
    print("Computing coherence values...")
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=bow_corpus, texts=documents, start=2, limit=40, step=6)

    # Show graph
    limit=40; start=2; step=6;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')



    """ --------------------------------------------------- LDA -------------------------------------------------------------- """


    print("\nWorking on simple LDA num_topics=16, passes=10...")


    lda_model_bow = gensim.models.LdaModel(bow_corpus, num_topics=16, id2word=dictionary, passes=10)

    
    for idx, topic in lda_model_bow.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))


    print('\nPerplexity: ', lda_model_bow.log_perplexity(bow_corpus))

    coherence_model_lda = CoherenceModel(model=lda_model_bow, texts=documents, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


    print('\nworking on topic visualization')
    
    vis = pyLDAvis.gensim.prepare(lda_model_bow, bow_corpus, dictionary)
    pyLDAvis.save_html(vis,'LDA_visualized.html')

    print('Time for this WHOLE thing: {} mins'.format(round((time() - t) / 60, 2)))
    plt.show()

if __name__ == "__main__":
    main()
