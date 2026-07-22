import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import pandas as pd
import os
from pandas.api.types import CategoricalDtype

df = pd.read_csv("../data/processed/csgo_cleaned_2.csv")

df2 = df.drop(columns=['file', 'date', 'tick', 'seconds', 'award', 'att_id', 'vic_id', 'ct_eq_val', 't_eq_val', 'map_1'])
df2["inbetween_distance"] = np.sqrt(np.square(df2["att_pos_x"] - df2["vic_pos_x"]) + np.square(df2["att_pos_y"] - df2["vic_pos_y"]))

df2["bombsite_a_x"] = np.where(df["map"] == "de_mirage", -390, np.nan)
df2["bombsite_a_y"] = np.where(df["map"] == "de_mirage", -2090, np.nan)
df2["bombsite_b_x"] = np.where(df["map"] == "de_mirage", -2160, np.nan)
df2["bombsite_b_y"] = np.where(df["map"] == "de_mirage", -280, np.nan)
df2["bombsite_a_att_distance"] = np.sqrt(np.square(df2["att_pos_x"] - df2["bombsite_a_x"]) + np.square(df2["att_pos_y"] - df2["bombsite_a_y"]))
df2["bombsite_b_att_distance"] = np.sqrt(np.square(df2["att_pos_x"] - df2["bombsite_b_x"]) + np.square(df2["att_pos_y"] - df2["bombsite_b_y"]))
df2["bombsite_a_vic_distance"] = np.sqrt(np.square(df2["vic_pos_x"] - df2["bombsite_a_x"]) + np.square(df2["vic_pos_y"] - df2["bombsite_a_y"]))
df2["bombsite_b_vic_distance"] = np.sqrt(np.square(df2["vic_pos_x"] - df2["bombsite_b_x"]) + np.square(df2["vic_pos_y"] - df2["bombsite_b_y"]))
df2["att_distance_to_bombsite"] = np.where(df2["map"] == "de_mirage", np.minimum(df2["bombsite_a_att_distance"], df2["bombsite_b_att_distance"]), np.nan)
df2["vic_distance_to_bombsite"] = np.where(df2["map"] == "de_mirage", np.minimum(df2["bombsite_a_vic_distance"], df2["bombsite_b_vic_distance"]), np.nan)
df3 = df2.drop(columns=["bombsite_a_att_distance", "bombsite_b_att_distance", "bombsite_a_vic_distance", "bombsite_b_vic_distance",
                       "bombsite_a_x", "bombsite_a_y", "bombsite_b_x", "bombsite_b_y"])

df3["total_dmg"] = df3["hp_dmg"] + df3["arm_dmg"]
df3["is_headshot"] = df3["hitbox"] == "Head"
df4 = df3.drop(columns=["hp_dmg", "arm_dmg", "hitbox"])

tier_boundaries = [0, 6, 10, 14, 18]
tiers = ["Silver", "Gold Nova", "Master Guardian", "Top Four"]

df4["att_tier"] = pd.cut(df4["att_rank"], bins=tier_boundaries, labels=tiers)
df4["vic_tier"] = pd.cut(df4["vic_rank"], bins=tier_boundaries, labels=tiers)
df5 = df4.drop(columns=["att_rank", "vic_rank"])
df5["att_tier"].describe()

# sorted_data = [df5.loc[df5["att_tier"] == "Silver", "total_dmg"], df5.loc[df5["att_tier"] == "Gold Nova", "total_dmg"], 
#     df5.loc[df5["att_tier"] == "Master Guardian", "total_dmg"], df5.loc[df5["att_tier"] == "Top Four", "total_dmg"]]
# labels = ['Silver', 'Gold Nova', 'Master Guardian', 'Top Four']

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.boxplot(sorted_data, patch_artist=True, tick_labels=labels)
# ax.set_title('att_tier vs. total_dmg')
# ax.set_ylabel('Total DMG')
# ax.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig("../images/04_total_dmg.png")
# plt.show()

df6 = df5[df5["total_dmg"] > 0]

# ct = pd.crosstab(df6["is_bomb_planted"], df6["att_tier"])
# ct

# ct = pd.crosstab(df6["round_type"], df6["att_tier"])
# ct

# ct = pd.crosstab(df6["wp"], df6["att_tier"])
# ct

# sorted_data = [df6.loc[df6["att_tier"] == "Silver", "inbetween_distance"], df6.loc[df6["att_tier"] == "Gold Nova", "inbetween_distance"], 
#     df6.loc[df6["att_tier"] == "Master Guardian", "inbetween_distance"], df6.loc[df6["att_tier"] == "Top Four", "inbetween_distance"]]
# labels = ['Silver', 'Gold Nova', 'Master Guardian', 'Top Four']

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.boxplot(sorted_data, patch_artist=True, tick_labels=labels)
# ax.set_title('att_tier vs. inbetween_distance')
# ax.set_ylabel('Inbetween Distance')
# ax.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig("../images/04_inbetween_distance.png")
# plt.show()

folder_path = '../data/processed'
file_name = 'csgo_cleaned_3.csv'
full_path = os.path.join(folder_path, file_name)

df6.to_csv(full_path, index=False)
print(f"File successfully saved to: {full_path}")