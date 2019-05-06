# -*- coding: utf-8 -*-

import os
import MeCab
from torchtext.data import Dataset, Example, Field
from torchtext.vocab import Vectors

tagger = MeCab.Tagger()
tagger.parse('')  # obtained https://github.com/SamuraiT/mecab-python3/issues/3


def tokenize(text):
    node = tagger.parseToNode(text)
    ret = []
    while node:
        if node.stat not in (2, 3):
            ret.append(node.surface)
        node = node.next
    return ret


def load_data(data_dir, emb_file):
    text_field = Field(sequential=True, tokenize=tokenize)
    label_field = Field(sequential=False, unk_token=None)
    fields = [('text', text_field), ('label', label_field)]
    examples = []

    for entry in os.scandir(data_dir):
        if entry.is_file():
            continue

        label = entry.name

        for doc_file in os.scandir(entry.path):
            if doc_file.name.startswith(label):
                with open(doc_file.path) as f:
                    text = '\n'.join(f.read().splitlines()[2:])
                    example = Example.fromlist([text, label], fields)
                    examples.append(example)

    data = Dataset(examples, fields)

    (train_data, test_data) = data.split(0.7)

    text_field.build_vocab(train_data)
    label_field.build_vocab(data)

    vectors = Vectors(emb_file)
    text_field.vocab.load_vectors(vectors)

    return train_data, test_data, text_field, label_field
