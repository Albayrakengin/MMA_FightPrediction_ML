import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score

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

#Filling reach differential

fighter_names = df["fighter"]
fighter_reach_diff = np.array([])
for i in range(0, len(df['fighter']), 2):
    fighter_reach = float(df.iloc[i]['reach'])
    opponent_reach = float(df.iloc[i+1]['reach'])
    fighter_reach_diff = np.append(fighter_reach_diff, fighter_reach/opponent_reach)
    fighter_reach_diff = np.append(fighter_reach_diff, opponent_reach/fighter_reach)
    
df['reach_differential'] = fighter_reach_diff

#Filling height differential

fighter_height_diff = np.array([])
for i in range(0, len(df['fighter']), 2):
    fighter_height = float(df.iloc[i]['height'])
    opponent_height = float(df.iloc[i+1]['height'])
    fighter_height_diff = np.append(fighter_height_diff, fighter_height/opponent_height)
    fighter_height_diff = np.append(fighter_height_diff, opponent_height/fighter_height)
    
df['height_differential'] = fighter_height_diff

#Filling age differential

fighter_age_diff = np.array([])
for i in range(0, len(df['fighter']), 2):
    fighter_age = float(df.iloc[i]['age'])
    opponent_age = float(df.iloc[i+1]['age'])
    fighter_age_diff = np.append(fighter_age_diff, fighter_age/opponent_age)
    fighter_age_diff = np.append(fighter_age_diff, opponent_age/fighter_age)
    
df['age_differential'] = fighter_age_diff

#Encoding the string variables

label_encoder = preprocessing.LabelEncoder()
df['fighter']= label_encoder.fit_transform(df['fighter'])
df["opponent"] = label_encoder.transform(df["opponent"])
df['stance'] = label_encoder.fit_transform(df["stance"])
df['method'] = label_encoder.fit_transform(df["method"])
df['division'] = label_encoder.fit_transform(df["division"])

sample = np.array([])
sample = label_encoder.fit_transform(["Ilia Topuria"])
print(sample)
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
df = df.drop(df.loc[:, 'knockdowns_differential':'ground_strikes_attempts_per_min'].columns, axis=1, inplace= False)
df = df.drop(df.loc[:, 'knockdowns':'ground_strikes_def'].columns, axis=1, inplace= False)
df = df.drop(df.loc[:, 'method':'time'].columns, axis=1, inplace= False)
print(df.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25, random_state=101)



rf_Model = RandomForestClassifier()

rf_Model.fit(X_train, Y_train)
print(rf_Model.score(X_test, Y_test))

predictions = rf_Model.predict(X_test)

# Compute precision, recall, and F1-score
precision = precision_score(Y_test, predictions)
recall = recall_score(Y_test, predictions)
f1 = f1_score(Y_test, predictions)

# Print the scores
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

