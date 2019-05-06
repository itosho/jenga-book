import gensim

model = gensim.models.KeyedVectors.load('jawiki_word2vec.bin')
model.wv.save_word2vec_format('jawiki_word2vec.txt')
