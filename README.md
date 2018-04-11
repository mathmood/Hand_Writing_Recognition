# Hand Writing Recognition
Hand Writing Recognition is one of significant applications of machine learning. Post services for example use this technology to sort and deliver our packages. In this simple project we train a neural network to recognize integers written by hand. The training set is provided by Dr Andrew Ng in his Coursera course and is available online. The data set includes 5000 training examples in X.csv which are accordingly labeled in y.csv, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image. Note that integer "0" is labeled by "10". 

Towards a machine learning algorithm, we wrote the following code. 

First, we call the required modules. 
```
import numpy as np
import pandas as pd
import scipy as sc
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
```

