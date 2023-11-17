import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import sklearn.model_selection
import sklearn.linear_model

data = pd.read_csv("./raceDataTest.csv").dropna()
X = (data.loc[:, data.columns != 'NextLapTime']).to_numpy()
Y = (data.loc[:, data.columns == 'NextLapTime']).to_numpy()

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, shuffle=True,)

perceptron = sklearn.linear_model.Perceptron()
perceptron.fit(X_train, Y_train.ravel())
Y_predict = perceptron.predict(X_test)

error = np.sqrt(np.mean((Y_test-Y_predict)**2))
print("The model's error in seconds is: ", error/(10**6))