from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string
from gensim.models.phrases import Phrases, Phraser
from time import time

tokenized_sentences = []

f = open("train.ft.txt",'r',encoding="utf8")

data = open("tokens.txt","a")

i = 0

for line in f:    
    s = line.split()
    tokenized_sentences.append(s[1:])   
    print("Working...", i)
    i = i+1
    if i == 10000: #ennyi sor
        break
    
f.close()

#Phrasing ,bigraming and building sentences
"""
phrases = Phrases(tokenized_sentences, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[tokenized_sentences]
"""
#instantiating model
model = Word2Vec(min_count=20,
                     window=2,
                     size=10,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers = 3,
                     max_vocab_size = 50000)
#building vocab
model.build_vocab(tokenized_sentences, progress_per=10000)
#training
t = time()
model.train(tokenized_sentences, total_examples=model.corpus_count, epochs=1, report_delay=1)
print('Time to train model: {} mins'.format(round((time() - t) / 60, 2)))
#saving model
model.save("word2vec.model")
