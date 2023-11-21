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



ridges = [10,1,0.1,0.01,0.001,0.0001,0.00001]
errors = []

for r in ridges:

    error = 0 

    for i in range(50):

        groups = Baseline + Identity + Tyre +  SectorTimes  + SessionTimes + RecordTimes + LapStatus

        totalFrame = pd.read_csv("./raceDataTest.csv").dropna()
        data = totalFrame[groups]

        X = (data.loc[:, data.columns != 'NextLapTime']).to_numpy()
        Y = (data.loc[:, data.columns == 'NextLapTime']).to_numpy()

        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y.ravel(), test_size=0.2, shuffle=True,)
        X_train, X_eval, Y_train, Y_eval = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.25, shuffle=True,)
        #X_train = sklearn.preprocessing.normalize(X_train)
        #X_test = sklearn.preprocessing.normalize(X_test)

        perceptron = sklearn.linear_model.Ridge(alpha = r, tol=0.000001)
        perceptron.fit(X_train, Y_train)
        Y_predict = perceptron.predict(X_test)
        Y_eval_predict = perceptron.predict(X_eval)

        error += np.sqrt(np.mean((Y_eval-Y_eval_predict)**2))

    errors.append(error/50)


plt.title("Ridge Size Vs Eval Error")
plt.xlabel("Ridge Size")
plt.ylabel("Error")
plt.semilogx(ridges,errors)
plt.show()