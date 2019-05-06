# -*- coding: utf-8 -*-

from torch.nn import Module, EmbeddingBag, Linear, Parameter


class NBoW(Module):

    def __init__(self, class_size, vectors):
        super(NBoW, self).__init__()
        self.nbow_layer = EmbeddingBag(vectors.size(0), vectors.size(1))
        self.nbow_layer.weight = Parameter(vectors)
        self.output_layer = Linear(vectors.size(1), class_size)

    def forward(self, words):
        feature_vector = self.nbow_layer(words)
        return self.output_layer(feature_vector)
