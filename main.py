import numpy as np
import pandas as pd

"""
        Planning to keep inplace False in order to prevent undesired changes.
"""
df = pd.read_csv("MMA_FightPrediction_ML\masterdataframe.csv")

irrelevant_columns = np.array(["event_url", "fight_url", "fighter_url", "opponent_url", "referee", "time_format"])

for i in irrelevant_columns:
    df = df.drop(i, axis= 1, inplace=False)

print(df.head())