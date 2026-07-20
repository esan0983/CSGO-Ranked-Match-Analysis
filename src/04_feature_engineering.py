import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import pandas as pd
import os

df = pd.read_csv("../data/processed/csgo_cleaned_2.csv")

df2 = df.drop(columns=['file', 'date', 'tick', 'seconds', 'award', 'att_id', 'vic_id', 'ct_eq_val', 't_eq_val', 'map_1'])
df2["inbetween_distance"] = np.sqrt(np.square(df2["att_pos_x"] - df2["vic_pos_x"]) + np.square(df2["att_pos_y"] - df2["vic_pos_y"]))

folder_path = '../data/processed'
file_name = 'csgo_cleaned_3.csv'
full_path = os.path.join(folder_path, file_name)

df2.to_csv(full_path, index=False)
print(f"File successfully saved to: {full_path}")