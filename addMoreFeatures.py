#Trains basic perceptron on everything and find weights 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing

Baseline1 = ["Event","LapTime", "NextLapTime","LapNumber", "PLapTime", "PPLapTime", "PPPLapTime"]
Identity = ["DriverNumber", "Team"] 
Tyre = ["Compound", "TyreLife", "FreshTyre"] 
SectorTimes = ["S1Time", "S2Time", "S3Time"] 
SessionTimes = ["S1STime", "S2STime", "S2STime"] 
RecordTimes = ["SpeedI2", "SpeedFL", "SpeedST"] 
LapStatus = ["IsPersonalBest", "Position", "Deleted", "TrackStatus"]



errors = []

for j in range(0,3):

    error = 0 

    for i in range(10):

        groups = Baseline1 + Identity + Tyre +  SectorTimes  + SessionTimes + RecordTimes + LapStatus
        totalFrame = pd.read_csv("./raceData2019MultiplePrevLaps.csv").dropna()
        data = totalFrame[groups]

        X = (data.loc[:, data.columns != 'NextLapTime']).to_numpy()
        Y = (data.loc[:, data.columns == 'NextLapTime']).to_numpy()

        X = sklearn.preprocessing.PolynomialFeatures(degree=j).fit_transform(X)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, shuffle=True,)
        X_train, X_eval, Y_train, Y_eval = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.25, shuffle=True,)
        #

        perceptron = sklearn.linear_model.Ridge(tol=0.00001)
        perceptron.fit(X_train, Y_train)
        Y_predict = perceptron.predict(X_test)
        Y_eval_predict = perceptron.predict(X_eval)

        error += np.sqrt(np.mean((Y_eval-Y_eval_predict)**2))

    errors.append(error/10)


plt.title("Error Vs Degree of Polynomial Features")
plt.xlabel("Degree of Polynomial Features")
plt.ylabel("Error")
plt.plot(range(0,3),errors)
plt.show()

print(errors)