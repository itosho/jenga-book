# -*- coding: utf-8 -*-
# python visualize.py jawiki_word2vec.bin word2vec_visualization.png

# notice: https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
import sys
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

# obtained https://github.com/ghmagazine/python_ml_book/blob/master/01/matplotlib_japanize/matplotlib_japanize.md
plt.rcParams["font.family"] = "IPAexGothic"

model = Word2Vec.load(sys.argv[1])

words = [word for (word, score) in model.wv.most_similar('錦糸町', topn=50)]
words.append('錦糸町')
vectors = np.vstack([model[word] for word in words])

tsne = TSNE(n_components=2)
Y = tsne.fit_transform(vectors)

x_coords = Y[:, 0]
y_coords = Y[:, 1]
plt.scatter(x_coords, y_coords)

for (word, x, y) in zip(words, x_coords, y_coords):
    plt.annotate(word, xy=(x, y), xytext=(5, -10), textcoords='offset points')

plt.savefig(sys.argv[2])
