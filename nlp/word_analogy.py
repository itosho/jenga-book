# -*- coding: utf-8 -*-
# python word_analogy.py jawiki_word2vec.bin オフサイド 野球 サッカー

import sys
import numpy as np
from gensim.models import Word2Vec

model_file = sys.argv[1]
(pos1, pos2, neg) = sys.argv[2:]

model = Word2Vec.load(model_file)
model.init_sims(replace=True)

vec = model[pos1] + model[pos2] - model[neg]
emb = model.wv.vectors_norm
sims = np.dot(emb, vec)

for index in np.argsort(-sims):
    word = model.wv.index2word[index]
    if word not in (pos1, pos2, neg):
        print('result:', word)
        break
