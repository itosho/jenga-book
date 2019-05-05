# -*- coding: utf-8 -*-
# python visualize.py jawiki_word2vec.bin word2vec_visualization.png

import sys
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

model = Word2Vec.load(sys.argv[1])

words = [word for (word, score) in model.wv.most_similar('砂糖', topn=50)]
words.append('砂糖')
vectors = np.vstack([model[word] for word in words])

tsne = TSNE(n_components=2)
Y = tsne.fit_transform(vectors)

x_coords = Y[:, 0]
y_coords = Y[:, 1]
plt.scatter(x_coords, y_coords)

for (word, x, y) in zip(words, x_coords, y_coords):
    plt.annotate(word, xy=(x, y), xytext=(5, -10), textcoords='offset points')

plt.savefig(sys.argv[2])
