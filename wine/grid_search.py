from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

wine = load_wine()
data = wine.data
target = wine.target

parameters = {
    'n_estimators': [3, 5, 10, 30, 50, 100],
    'max_features': [1, 3, 5, 10],
    'random_state': [0],
    'min_samples_split': [3, 5, 10, 30, 50],
    'max_depth': [3, 5, 10, 30, 50]
}

clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=5, iid=False)
clf.fit(data, target)

print(clf.best_estimator_)
