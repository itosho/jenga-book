# -*- coding: utf-8 -*-

import requests
from sklearn.datasets import load_wine

wine = load_wine()
data = wine.data
target = wine.target

url = 'http://127.0.0.1:5000/'

wine = data[0]
label = target[0]

post_data = {'wine': wine.tolist()}
response = requests.post(url, json=post_data).json()

print('correct: %d' % label)
print('result: %d' % response['label'])
