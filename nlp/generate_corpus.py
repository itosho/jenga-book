# -*- coding: utf-8 -*-
# python generate_corpus.py nif_context_ja.ttl.bz2 > jawiki_corpus.txt

import bz2
import sys
import MeCab
from rdflib import Graph

tagger = MeCab.Tagger('')
tagger.parse('')  # obtained https://github.com/SamuraiT/mecab-python3/issues/3


def read_ttl(f):
    while True:
        lines = [line.decode("utf-8").rstrip() for line in f.readlines(102400)]
        if not lines:
            break

        for triple in parse_lines(lines):
            yield triple


def parse_lines(lines):
    g = Graph()
    g.parse(data='\n'.join(lines), format='n3')
    return g


def tokenize(text):
    node = tagger.parseToNode(text)
    while node:
        if node.stat not in (2, 3):
            yield node.surface
        node = node.next


with bz2.BZ2File(sys.argv[1]) as in_file:
    for (_, p, o) in read_ttl(in_file):
        if p.toPython() == 'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString':
            for line in o.toPython().split('\n'):
                words = list(tokenize(line))
                if len(words) > 20:
                    print(' '.join(words))
