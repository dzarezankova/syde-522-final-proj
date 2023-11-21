#Trains basic perceptron on everything and find weights 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing

Baseline1 = ["Event","LapTime", "NextLapTime","LapNumber"]
Baseline2 = ["PLapTime"]
Baseline3 = ["PPLapTime"]
Baseline4 = ["PPPLapTime"]
Identity = ["DriverNumber", "Team"] 
Tyre = ["Compound", "TyreLife", "FreshTyre"] 
SectorTimes = ["S1Time", "S2Time", "S3Time"] 
SessionTimes = ["S1STime", "S2STime", "S2STime"] 
RecordTimes = ["SpeedI2", "SpeedFL", "SpeedST"] 
LapStatus = ["IsPersonalBest", "Position", "Deleted", "TrackStatus"]



errors = []

for j in range(4):

    error = 0 

    for i in range(100):

        groups = Baseline1 + Identity + Tyre +  SectorTimes  + SessionTimes + RecordTimes + LapStatus + (Baseline2 if i > 0 else []) + (Baseline3 if i > 1 else [])  + (Baseline4 if i > 2 else []) 

        totalFrame = pd.read_csv("./raceData2019MultiplePrevLaps.csv").dropna()
        data = totalFrame[groups]

        X = (data.loc[:, data.columns != 'NextLapTime']).to_numpy()
        Y = (data.loc[:, data.columns == 'NextLapTime']).to_numpy()

        #X = sklearn.preprocessing.normalize(X)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, shuffle=True,)
        X_train, X_eval, Y_train, Y_eval = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.25, shuffle=True,)
        #

        perceptron = sklearn.linear_model.Ridge()
        perceptron.fit(X_train, Y_train)
        Y_predict = perceptron.predict(X_test)
        Y_eval_predict = perceptron.predict(X_eval)

        error += np.sqrt(np.mean((Y_eval-Y_eval_predict)**2))

    errors.append(error/100)


plt.title("Error Vs Number of Previous Laps")
plt.xlabel("Number of Previous Laps")
plt.ylabel("Error")
plt.plot(range(4),errors)
plt.show()