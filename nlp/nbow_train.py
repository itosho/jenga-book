# -*- coding: utf-8 -*-
# python 

import sys
import torch
import torch.nn.functional as F
from torchtext.data import Iterator
from . import dataset
from . import nbow_model


def train(dataset_dir, emb_file, epoch, batch_size):
    (train_data, test_data, text_field, label_field) = dataset.load_data(dataset_dir, emb_file)

    class_size = len(label_field.vocab)

    nbow = nbow_model.NBoW(class_size, text_field.vocab.vectors)
    nbow.train()

    optimizer = torch.optim.Adam(nbow.parameters())

    train_iter = Iterator(train_data, batch_size)
    for n in range(epoch):
        for batch in train_iter:
            optimizer.zero_grad()

            logit = nbow(batch.text.t())
            loss = F.cross_entropy(logit, batch.label)
            loss.backward()

            optimizer.step()

        nbow.eval()

        (accuracy, num_correct) = compute_accuracy(nbow, test_data)
        print('Epoch: {} Accuracy: {:.2f}% ({}/{})'.format(n + 1, accuracy * 100, num_correct, len(test_data)))

        nbow.train()


def compute_accuracy(model, test_data):
    test_size = len(test_data)
    test_data = next(iter(Iterator(test_data, test_size)))

    logit = model(test_data.text.t())

    num_correct = (torch.max(logit, 1)[1].view(test_size) == test_data.label).sum().item()
    accuracy = float(num_correct) / test_size
    return accuracy, num_correct


if __name__ == '__main__':
    train(dataset_dir=sys.argv[1], emb_file=sys.argv[2], epoch=int(sys.argv[3]),
          batch_size=int(sys.argv[4]))
