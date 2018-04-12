import numpy as np
import pandas as pd
import scipy as sc
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


df = pd.read_csv("datavalidation.csv", index_col=0)

X = np.asarray(df.index)
Y = np.asarray(df.columns)
Z = df.as_matrix()

print(Y, X)
fig, ax = plt.subplots()

fig = plt.contourf(Y, X, Z, 20, cmap='RdGy')
fig = plt.colorbar();

ax.set_xticks(Y)
ax.set_xticklabels(Y)
ax.set_yticks(X)
ax.set_yticklabels(X)

ax.set_xlabel('alpha in the form of j: (10**((-1)*j))')
ax.set_ylabel('number of hidden layers')

plt.show(block=True)
plt.interactive(False)