#Trains basic perceptron on everything and find weights 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import sklearn.model_selection
import sklearn.linear_model


Baseline = ["LapTime", "NextLapTime","LapNumber"]
Identity = ["DriverNumber", "Team"] 
Tyre = ["Compound", "TyreLife", "FreshTyre"] 
SectorTimes = ["S1Time", "S2Time", "S3Time"] 
SessionTimes = ["S1STime", "S2STime", "S2STime"] 
RecordTimes = ["SpeedI2", "SpeedFL", "SpeedST"] 
LapStatus = ["IsPersonalBest", "Position", "Deleted", "TrackStatus"]


errors =  np.zeros(64)

for i in range(100):

    ind = 0


groups = Baseline + Identity + Tyre + SectorTimes  + SessionTimes + RecordTimes  + LapStatus 

totalFrame = pd.read_csv("./raceDataTest.csv").dropna()
data = totalFrame[groups]

X = (data.loc[:, data.columns != 'NextLapTime']).to_numpy()
Y = (data.loc[:, data.columns == 'NextLapTime']).to_numpy()

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y.ravel(), test_size=0.2, shuffle=True,)
perceptron = sklearn.linear_model.LinearRegression()
perceptron.fit(X_train, Y_train)
Y_predict = perceptron.predict(X_test)

error = np.sqrt(np.mean((Y_test-Y_predict)**2))
#print(error, " ", groups)
errors[ind] += round(error,2)
ind = ind + 1

weights = perceptron.coef_
print(weights)
