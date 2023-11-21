#Trains basic perceptron on everything and find weights 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing

Baseline = ["LapTime", "NextLapTime","LapNumber"]
Identity = ["DriverNumber", "Team"] 
Tyre = ["Compound", "TyreLife", "FreshTyre"] 
SectorTimes = ["S1Time", "S2Time", "S3Time"] 
SessionTimes = ["S1STime", "S2STime", "S2STime"] 
RecordTimes = ["SpeedI2", "SpeedFL", "SpeedST"] 
LapStatus = ["IsPersonalBest", "Position", "Deleted", "TrackStatus"]

errors =  0
weights = np.zeros(20)

for i in range(100):

    groups = Baseline + Identity + Tyre + SectorTimes  + SessionTimes + RecordTimes  + LapStatus 

    totalFrame = pd.read_csv("./raceDataTest.csv").dropna()
    data = totalFrame[groups]
    print(data.head())

    X = (data.loc[:, data.columns != 'NextLapTime']).to_numpy()
    Y = (data.loc[:, data.columns == 'NextLapTime']).to_numpy()

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y.ravel(), test_size=0.2, shuffle=True,)

    X_train = sklearn.preprocessing.normalize(X_train)
    X_test = sklearn.preprocessing.normalize(X_test)

    perceptron = sklearn.linear_model.LinearRegression()
    perceptron.fit(X_train, Y_train)
    Y_predict = perceptron.predict(X_test)

    error = np.sqrt(np.mean((Y_test-Y_predict)**2))
    errors += error

    weights += np.array(perceptron.coef_)
    

print(errors/100)
print(weights/100)