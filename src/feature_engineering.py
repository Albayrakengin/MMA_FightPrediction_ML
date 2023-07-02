import numpy as np
from sklearn import preprocessing

def engineer_features(df):
    
    #Encoding the string variables

    label_encoder = preprocessing.LabelEncoder()
    df['fighter']= label_encoder.fit_transform(df['fighter'])
    df["opponent"] = label_encoder.transform(df["opponent"])
    df['stance'] = label_encoder.fit_transform(df["stance"])
    df['method'] = label_encoder.fit_transform(df["method"])
    df['division'] = label_encoder.fit_transform(df["division"])

    
    # Convert time to seconds
    time_array = np.array([])
    for i in df['time']:
        time_str = i
        minutes, seconds = time_str.split(":")
        minutes = int(minutes)
        seconds = int(seconds) 
        time_array = np.append(time_array, ((minutes * 60) + seconds))
    df['time'] = time_array
    
    return df