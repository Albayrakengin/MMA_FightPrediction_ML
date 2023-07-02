import numpy as np

def preprocess_data(df):
    # Drop irrelevant columns
    irrelevant_columns = np.array(["event_url", "fight_url", "fighter_url", "opponent_url", "referee", "time_format", "dob", "date"])
    df = df.drop(irrelevant_columns, axis=1)
    
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
    

    
    return df