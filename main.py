import numpy as np
import pandas as pd


# Planning to keep inplace False in order to prevent undesired changes.
df = pd.read_csv("MMA_FightPrediction_ML\masterdataframe.csv")
irrelevant_columns = np.array(["event_url", "fight_url", "fighter_url", "opponent_url", "referee", "time_format", "dob"])

for i in irrelevant_columns:
    df = df.drop(i, axis= 1, inplace=False)
df = df.drop(df.loc[:, 'avg_knockdowns':'precomp_recent_avg_ground_strikes_attempts_per_min'].columns, axis=1, inplace= False)

#Filling missing values.

#Since the fighters were way older back in the days compare to now, I will be filling them separetly
array = df['age'].to_numpy()[0:500]
array = array[~np.isnan(array)]
numOfNan = 500 - len(array)
arrayAverage = int(array.mean())

df["age"] = df["age"].fillna(arrayAverage,  limit=numOfNan)

#Filling rest of the age
array = df['age'].to_numpy()[500:]
array = array[~np.isnan(array)]
numOfNan = (13323 - 500) - len(array)
arrayAverage = int(array.mean())
df["age"] = df["age"].fillna(arrayAverage,  limit=numOfNan)

#print(df.head())