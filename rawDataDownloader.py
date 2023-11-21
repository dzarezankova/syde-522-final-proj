import numpy as np 
import pandas as pd 
import fastf1

def timeToMs(time):
    return 0.0 + time.seconds + time.microseconds/1000000

def CompoundToNumber(tyre):
    if (tyre == "SOFT"):
        return 1 
    elif (tyre == "MEDIUM"):
        return 2 
    elif (tyre == "HARD"):
        return 3 
    elif (tyre == "INTERMEDIATE"):
        return 4 
    elif (tyre == "WET"):
        return 5 
    else: 
        return 6
    
def toNumber(tf):
    if (tf == "True"):
        return 1 
    else: 
        return 0

teamList = []

def teamToNumber(tf):
    if tf in teamList:
        return teamList.index(tf)
    else:
        teamList.append(tf)
        return teamList.index(tf)

session = fastf1.get_session(2019,  "monza", 'R')
session.load()
laps = session.laps

laps.to_csv("./laps.csv")

Baseline = ["LapTime", "NextLapTime","LapNumber"]
Identity = ["DriverNumber", "Team"] 
Tyre = ["Compound", "TyreLife", "FreshTyre"] 
SectorTimes = ["S1Time", "S2Time", "S3Time"] 
SessionTimes = ["S1STime", "S2STime", "S2STime"] 
RecordTimes = ["SpeedI2", "SpeedFL", "SpeedST"] 
LapStatus = ["IsPersonalBest", "Position", "Deleted", "TrackStatus"]

cols = Baseline + Identity + Tyre + SectorTimes + SessionTimes + RecordTimes + LapStatus
raceData = pd.DataFrame(columns = cols)

index = 0 

for i in range(len(laps)-1):
    if (laps.loc[i+1,"DriverNumber"] == laps.loc[i,"DriverNumber"]):

        #baseline 
        LapTime = timeToMs(laps.loc[i,"LapTime"])
        NextLapTime = timeToMs(laps.loc[i+1, "LapTime"])
        LapNumber = laps.loc[i,"LapNumber"]

        #Identity 
        DriverNumber = laps.loc[i,"DriverNumber"]
        Team = teamToNumber(laps.loc[i,"Team"])

        #Tyre 
        Compound = CompoundToNumber(laps.loc[i,"Compound"])
        TyreLife = laps.loc[i,"TyreLife"]
        FreshTyre = toNumber(laps.loc[i,"FreshTyre"])
        
        #Sector Times
        S1Time = timeToMs(laps.loc[i,"Sector1Time"])
        S2Time = timeToMs(laps.loc[i,"Sector2Time"])
        S3Time = timeToMs(laps.loc[i,"Sector3Time"])

        #session Times 
        S1STime = timeToMs(laps.loc[i,"Sector1SessionTime"])
        S2STime = timeToMs(laps.loc[i,"Sector2SessionTime"])
        S3STime = timeToMs(laps.loc[i,"Sector3SessionTime"])

        #record times 
        SpeedI2 = laps.loc[i,"SpeedI2"]
        SpeedFL = laps.loc[i,"SpeedFL"]
        SpeedST = laps.loc[i,"SpeedST"]

        #Lap Status 
        IsPersonalBest = toNumber(laps.loc[i,"IsPersonalBest"])
        Position = laps.loc[i,"Position"]
        Deleted = toNumber(laps.loc[i,"Deleted"])
        TrackStatus = laps.loc[i,"TrackStatus"]

        raceData.loc[index] = ({"LapTime":LapTime,"NextLapTime":NextLapTime, "LapNumber": LapNumber, "DriverNumber":DriverNumber,"Team": Team, "Compound":Compound,
                                "TyreLife":TyreLife,"FreshTyre":FreshTyre, "S1Time": S1Time, "S2Time":S2Time,"S3Time": S3Time, "S1STime":S1STime,
                                "S2STime":S2STime,"S3STime":S3STime, "SpeedI2": SpeedI2, "S2Time":S2Time,"SpeedFL": SpeedFL, "SpeedST":SpeedST,
                                "IsPersonalBest":IsPersonalBest,"Position":Position, "Deleted": Deleted, "TrackStatus":TrackStatus})
        index = index + 1 

 

raceData.to_csv("./raceDataTest.csv")


