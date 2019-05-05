# -*- coding: utf-8 -*-

import logging
import multiprocessing
import sys
from gensim.models import word2vec

logging.basicConfig(level=logging.INFO)

cpu_count = multiprocessing.cpu_count()
model = word2vec.Word2Vec(
    word2vec.LineSentence(sys.argv[1]),
    sg=1,
    size=100,
    window=5,
    min_count=5,
    iter=5,
    workers=cpu_count
)
model.save(sys.argv[2])
