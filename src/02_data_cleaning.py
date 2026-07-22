import pandas as pd
import os

df = pd.read_csv("../data/raw/mm_master_demos.csv")

# We are removing actual named teams because we assume that the team chemistry is already set in stone,
# thus varying positioning greatly from those who solo queue.
df1 = df[df["att_team"].isin(["Team 1", "Team 2", "World"])]

# Same idea with victim team.
df2 = df1[df1["vic_team"].isin(["Team 1", "Team 2"])]

# Map data is only limited to seven maps (the most popular ones), so we will remove the rest.
map_list = ["de_cache", "de_cbble", "de_dust2", "de_inferno", "de_mirage", "de_overpass", "de_train"]
df3 = df2[df2["map"].isin(map_list)]

# Spelling error.
df3.loc[df['wp_type'] == 'Unkown', 'wp_type'] = 'Unknown'

# According to research, there is no "rank zero."
df4 = df3[(df3["att_rank"] > 0) & (df3["vic_rank"] > 0)]

# Everything else has been checked: missing values, duplicates, etc. There were no errors.
# Export:
folder_path = '../data/processed'
file_name = 'csgo_cleaned_1.csv'
full_path = os.path.join(folder_path, file_name)

df4.to_csv(full_path, index=False)
print(f"File successfully saved to: {full_path}")

# Load map data and check:
df = pd.read_csv("../data/raw/map_data.csv")

# Rename unnamed column for SQL purposes.
df.rename(columns={'Unnamed: 0': 'map'}, inplace=True)


# Now, map data is clean and there's nothing wrong.
folder_path = '../data/processed'
file_name = 'map_data.csv'
full_path = os.path.join(folder_path, file_name)

df.to_csv(full_path, index=False)
print(f"File successfully saved to: {full_path}")

# Now we just need to check validity of coordinates via SQL. Check map_check.sql.