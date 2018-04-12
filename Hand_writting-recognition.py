import numpy as np
import pandas as pd
import scipy as sc
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt




X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

df = pd.DataFrame()
for i in range(10,250,10):
    for j in range(-2,10):
        nn_alg = MLPClassifier(solver='lbfgs', alpha=10**((-1)*j), hidden_layer_sizes=(i,), random_state=1)
        nn_alg.fit(X_train, y_train)
        y_pred = nn_alg.predict(X_test)
        df.at[i,j] = nn_alg.score(X_test, y_test)

df.to_csv("datavalidation.csv")


nn_alg = MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(225,), random_state=1)
nn_alg.fit(X_train, y_train)
y_pred = nn_alg.predict(X_test)
print('Accuracy of the corresponding neural network classifier on test set: {:.2f}'.format(nn_alg.score(X_test, y_test)))
