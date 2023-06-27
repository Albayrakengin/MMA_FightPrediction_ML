import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Planning to keep inplace False in order to prevent undesired changes.
df = pd.read_csv("MMA_FightPrediction_ML\masterdataframe.csv")
irrelevant_columns = np.array(["event_url", "fight_url", "fighter_url", "opponent_url", "referee", "time_format", "dob", "date"])

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

#filling reach
array = df['reach'].to_numpy()
array = array[~np.isnan(array)]
arrayAverage = int(array.mean())
df["reach"] = df["reach"].fillna(arrayAverage)

#filling height
array = df['height'].to_numpy()
array = array[~np.isnan(array)]
arrayAverage = int(array.mean())
df["height"] = df["height"].fillna(arrayAverage)

#Encoding the string variables

label_encoder = preprocessing.LabelEncoder()
df['fighter']= label_encoder.fit_transform(df['fighter'])
df["opponent"] = label_encoder.transform(df["opponent"])
df['stance'] = label_encoder.fit_transform(df["stance"])
df['method'] = label_encoder.fit_transform(df["method"])
df['division'] = label_encoder.fit_transform(df["division"])

time_array = np.array([])
for i in df['time']:
    time_str = i
    minutes, seconds = time_str.split(":")
    minutes = int(minutes)
    seconds = int(seconds) 
    time_array = np.append(time_array, ((minutes * 60) + seconds))
df['time'] = time_array


#Splitting data into X and Y 

Y = df["result"]
X = df.drop("result", axis=1, inplace=False)

#Split data into train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25, random_state=101)

print(f"x train = {X_train.shape}")
print(f"y train = {Y_train.shape}")
print(f"x test = {X_test.shape}")
print(f"y test = {Y_test.shape}")


#print(df.head())