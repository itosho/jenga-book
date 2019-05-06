# -*- coding: utf-8 -*-
# python similar_words.py jawiki_word2vec.bin 野球

import sys
import numpy as np
from gensim.models import Word2Vec

COUNT = 3
model_file = sys.argv[1]
target_word = sys.argv[2]

model = Word2Vec.load(model_file)
model.init_sims(replace=True)

vec = model[target_word]
emb = model.wv.vectors_norm
sims = np.dot(emb, vec)

count = 0
for index in np.argsort(-sims):
    word = model.wv.index2word[index]
    if word != target_word:
        print('%s (similarity: %.2f)' % (word, sims[index]))
        count += 1
        if count == COUNT:
            break
