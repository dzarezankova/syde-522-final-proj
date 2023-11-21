#goal: Create a pandas data frame that can be used for lap time prediciton 
import numpy as np 
import pandas as pd 
import fastf1

#starting off with a basic model - the difficulty can easily be increased 
basic_cols = ["Event", "DriverNumber", "LapTime", "LapNumber", "Compound", "TyreLife", "NextLapTime"]
expanded_cols = ["Event", "DriverNumber", "LapTime", "LapNumber", "Compound", "TyreLife", "Weather","NextLapTime"]
raceData = pd.DataFrame(columns = basic_cols)

def getSessionsPerYear(year):
    s = fastf1.get_event_schedule(year) 
    locations = s["Location"].to_list()
    return locations 

def timeToMs(time):
    return time.microseconds 

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

years = [2019]
index = 0 

for j in range(len(years)):

    events = getSessionsPerYear(years[j])
    
    for k in range(len(events)):
        session = fastf1.get_session(years[j],  events[k], 'R')
        session.load()
        laps = session.laps

        for i in range(len(laps)-1):
            if (laps.loc[i+1,"DriverNumber"] == laps.loc[i,"DriverNumber"]):
                Event = k + j*100 
                DriverNumber = laps.loc[i,"DriverNumber"]
                LapTime = timeToMs(laps.loc[i,"LapTime"])
                LapNumber = laps.loc[i,"LapNumber"]
                Compound = CompoundToNumber(laps.loc[i,"Compound"])
                TyreLife = laps.loc[i,"TyreLife"]
                NextLapTime = timeToMs(laps.loc[i+1, "LapTime"])
                # Weather =

                raceData.loc[index] = ({"Event": Event, "DriverNumber":DriverNumber,"LapTime":LapTime, "LapNumber": LapNumber, "Compound":Compound,"TyreLife": TyreLife, "NextLapTime":NextLapTime})
                index = index + 1 

raceData.to_csv("./raceDataTest.csv")





    #hold up with next lap time cause its more complicated 


#print(laps.head())