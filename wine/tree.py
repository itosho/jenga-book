import pydot
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
data = wine.data
target = wine.target

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

print(clf.score(X_test, Y_test))

tree.export_graphviz(
    clf,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    out_file='tree.dot',
)

(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
