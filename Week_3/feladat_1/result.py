from gensim.models import Word2Vec


model = Word2Vec.load("word2vec.model")

vector = model.wv

result = vector.most_similar(positive=['good'])

print(result)
