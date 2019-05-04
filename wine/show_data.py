import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()

data_x = pd.DataFrame(data=data.data, columns=data.feature_names)
print(data_x.head())

data_y = pd.DataFrame(data=data.target)
data_y = data_y.rename(columns={0: 'class'})
print(data_y.head())
