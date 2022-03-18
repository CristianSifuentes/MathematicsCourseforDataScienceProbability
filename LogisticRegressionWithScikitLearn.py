from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd

atrib_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
X, y = load_iris(return_X_y=True)

X[:2]

clf = LogisticRegression(random_state=10, solver='liblinear').fit(X[:100], y[:100])

clf.coef_

model_coefs = pd.DataFrame(clf.coef_, columns=atrib_names)
model_coefs