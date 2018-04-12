# Hand Writing Recognition
Hand Writing Recognition is one of significant applications of machine learning. Post services for example use this technology to sort and deliver our packages. In this simple project we train a neural network to recognize integers written by hand. The training set is provided by Dr Andrew Ng in his Coursera course and is available online. The data set includes 5000 training examples in X.csv which are accordingly labeled in y.csv, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image. Note that integer "0" is labeled by "10". 

Towards a machine learning algorithm, we wrote the following code. 

First, we call the required modules. 
```
import numpy as np
import pandas as pd
import scipy as sc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
```

In the next step, we read our data and split it into training/test set to use it in the algorithms. 

```
X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=40)
```

First let us use a simple logistic regression. The accuracy of this algorithm is %91. 

```
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

>> Accuracy of logistic regression classifier on test set: 0.91
```
Now let us use neural network to improve the accuracy. To do so, we cross validate the data for different values of hidden layers and regularization.
```
df = pd.DataFrame()
for i in range(10,20):
    for j in range(-5,3):
        nn_alg = MLPClassifier(solver='lbfgs', alpha=10**((-1)*j), hidden_layer_sizes=(i,), random_state=1)
        nn_alg.fit(X_train, y_train)
        y_pred = nn_alg.predict(X_test)
        df.at[i,j] = nn_alg.score(X_test, y_test)
```
The datafram ```df``` can be easily studies through a simple plotting. 
```

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
```
The outcome is the attached file `Figure_1.png`. As one imediately notice, the most efficient values are `alpha = ???` and hidden layers ???. 
```
nn_alg = MLPClassifier(solver='lbfgs', alpha=???, hidden_layer_sizes=(???,), random_state=1)
nn_alg.fit(X_train, y_train)
y_pred = nn_alg.predict(X_test)
print('Accuracy of the corresponding neural network classifier on test set: {:.2f}'.format(nn_alg.score(X_test, y_test)))

>>
```
