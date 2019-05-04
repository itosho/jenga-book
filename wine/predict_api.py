import pickle
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

with open('model.pkl', mode='rb') as fp:
    clf = pickle.load(fp)


@app.route('/', methods=['POST'])
def predict():
    wine = request.json['wine']
    wine = np.array([wine])

    label = clf.predict(wine).tolist()

    return jsonify(dict(label=label[0]))


if __name__ == '__main__':
    app.run()
