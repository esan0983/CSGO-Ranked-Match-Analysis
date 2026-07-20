import duckdb
import pandas as pd
import os

maps = pd.read_csv("data/processed/map_data.csv")
csgo = pd.read_csv("data/processed/csgo_cleaned_1.csv")

valid = duckdb.sql(
"""
SELECT * FROM csgo
LEFT JOIN maps ON csgo.map == maps.map
WHERE att_pos_x >= StartX 
    AND att_pos_x <= EndX 
    AND att_pos_y >= StartY 
    AND att_pos_y <= EndY
    AND vic_pos_x >= StartX 
    AND vic_pos_x <= EndX 
    AND vic_pos_y >= StartY 
    AND vic_pos_y <= EndY;
""").df()

print(valid)

folder_path = 'data/processed'
file_name = 'csgo_cleaned_2.csv'
full_path = os.path.join(folder_path, file_name)

valid.to_csv(full_path, index=False)
print(f"File successfully saved to: {full_path}")