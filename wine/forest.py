from sklearn import ensemble
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
data = wine.data
target = wine.target

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=1)
clf = clf.fit(X_train, Y_train)

print(clf.score(X_test, Y_test))
