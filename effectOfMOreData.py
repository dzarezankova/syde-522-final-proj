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



wasteSize = [0.01,0.1,0.2,0.4,0.8,1]
errors = []

for w in wasteSize:

    error = 0 

    for i in range(100):

        groups = Baseline + Identity + Tyre +  SectorTimes  + SessionTimes + RecordTimes + LapStatus

        totalFrame = pd.read_csv("./raceData2019.csv").dropna()
        data = totalFrame[groups]

        X = (data.loc[:, data.columns != 'NextLapTime']).to_numpy()
        Y = (data.loc[:, data.columns == 'NextLapTime']).to_numpy()

        X, wasteX, Y, wasteY = sklearn.model_selection.train_test_split(X, Y.ravel(), test_size=w, shuffle=True,)
        X = sklearn.preprocessing.normalize(X)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, shuffle=True,)
        X_train, X_eval, Y_train, Y_eval = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.25, shuffle=True,)
        #

        perceptron = sklearn.linear_model.Ridge(alpha =1, tol=0.0001)
        perceptron.fit(X_train, Y_train)
        Y_predict = perceptron.predict(X_test)
        Y_eval_predict = perceptron.predict(X_eval)

        error += np.sqrt(np.mean((Y_eval-Y_eval_predict)**2))

    errors.append(error/100)


plt.title("Tolerance Size Vs Portion of Total Data in Use")
plt.xlabel("Percent of Data Set")
plt.ylabel("Error")
plt.semilogx(wasteSize,errors)
plt.show()