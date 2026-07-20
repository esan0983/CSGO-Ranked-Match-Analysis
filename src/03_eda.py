import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.image as mpimg

df = pd.read_csv("../data/processed/csgo_cleaned_2.csv")

# Uncomment what you want to see.

# MAP COUNTS
# df["map"].value_counts().plot(kind="bar")
# plt.title('Map Counts')
# plt.xlabel('Map')
# plt.ylabel('Count')
# plt.savefig("../images/03_map_counts.png")
# plt.show()

# ROUND COUNTS
# round_counts = df["round"].value_counts()
# round_counts.sort_index().plot(kind='bar')
# plt.xlabel('Round')
# plt.ylabel('Counts')
# plt.title('Round Counts')
# plt.savefig("../images/03_round_counts.png")
# plt.show()

# TICK DISTRIBUTION
# plt.boxplot(df["tick"])
# plt.title("Ticks")
# plt.savefig("../images/03_ticks.png")
# plt.show()

# SECONDS DISTRIBUTION
# plt.boxplot(df["seconds"])
# plt.title("Seconds")
# plt.savefig("../images/03_seconds.png")
# plt.show()

# ARMOR DAMAGE DISTRIBUTION
# plt.boxplot(df["arm_dmg"])
# plt.title("Armor Damage")
# plt.savefig("../images/03_arm_dmg.png")
# plt.show()

# HP DAMAGE DISTRIBUTION
# plt.boxplot(df["hp_dmg"])
# plt.title("HP Damage")
# plt.savefig("../images/03_hp_dmg.png")
# plt.show()

# BOMB PLANTED BOOL COUNTS
# df["is_bomb_planted"].value_counts().plot(kind="bar")
# plt.title('Is bomb planted?')
# plt.xlabel('True/False')
# plt.ylabel('Count')
# plt.savefig("../images/03_bomb_counts.png")
# plt.show()

# BOMB SITE COUNTS
# temp_df = df.loc[df["is_bomb_planted"] == True, "bomb_site"]
# temp_df.value_counts().plot(kind="bar")
# plt.title('Bomb Site Counts')
# plt.xlabel('Bomb Site')
# plt.ylabel('Count')
# plt.savefig("../images/03_bomb_site_counts.png")
# plt.show()

# HITBOX COUNTS
# df["hitbox"].value_counts().plot(kind="bar")
# plt.title('Hitbox Counts')
# plt.xlabel('Hitbox')
# plt.ylabel('Count')
# plt.savefig("../images/03_hitbox_counts.png")
# plt.show()

# WEAPON COUNTS
# df["wp"].value_counts().plot(kind="bar")
# plt.title('Weapon Counts')
# plt.xlabel('Weapon')
# plt.ylabel('Count')
# plt.savefig("../images/03_wp_counts.png")
# plt.show()

# WEAPON TYPE COUNTS
# df["wp_type"].value_counts().plot(kind="bar")
# plt.title('Weapon Type Counts')
# plt.xlabel('Weapon Type')
# plt.ylabel('Count')
# plt.savefig("../images/03_wp_type_counts.png")
# plt.show()

# ATTACKER RANK COUNTS
# att_rank_counts = df["att_rank"].value_counts()
# att_rank_counts.sort_index().plot(kind='bar')
# plt.xlabel('Attacker Rank')
# plt.ylabel('Counts')
# plt.title('Attacker Rank Counts')
# plt.savefig("../images/03_att_rank_counts.png")
# plt.show()

# VICTIM RANK COUNTS
# vic_rank_counts = df["vic_rank"].value_counts()
# vic_rank_counts.sort_index().plot(kind='bar')
# plt.xlabel('Victim Rank')
# plt.ylabel('Counts')
# plt.title('Victim Rank Counts')
# plt.savefig("../images/03_vic_rank_counts.png")
# plt.show()

# ROUND TYPE COUNTS
# df["round_type"].value_counts().plot(kind="bar")
# plt.title('Round Type Counts')
# plt.xlabel('Round Type')
# plt.ylabel('Count')
# plt.savefig("../images/03_round_type_counts.png")
# plt.show()

# AVG MATCH RANK COUNTS
# avg_rank_counts = df["avg_match_rank"].value_counts()
# avg_rank_counts.sort_index().plot(kind='bar')
# plt.xlabel('Average Match Rank')
# plt.ylabel('Counts')
# plt.title('Average Match Rank Counts')
# plt.savefig("../images/03_avg_rank_counts.png")
# plt.show()

# SOME CROSSTABS
# pd.crosstab(df["hitbox"], df["wp_type"], normalize=1)
# pd.crosstab(df["att_rank"], df["wp_type"], normalize=0)

# ATT_RANK VS WP RELATIONSHIP
# ct = pd.crosstab(df["att_rank"], df["wp"], normalize=0)
# sns.heatmap(ct, vmin=0, vmax=ct.values.max(), cmap="coolwarm")
# plt.savefig("../images/03_rank_wp_heatmap.png")
# plt.show()

# COORDINATE VISUALIZATIONS: EITHER ATTACKER POSITION OR VICTIM POSITION, YOU CAN MODIFY THE CODE
# img = mpimg.imread('../images/de_mirage.png')

# # REFER TO MAP DATA FOR THIS
# x_min = -3217
# x_max = 1912
# y_min = -3401
# y_max = 1682

# fig, ax = plt.subplots(figsize=(10,10))
# ax.imshow(img, extent=[x_min, x_max, y_min, y_max])  # your map's start/end coords
# sns.kdeplot(x=df.loc[df["map"] == "de_mirage", 'att_pos_x'], y=df.loc[df["map"] == "de_mirage", 'att_pos_y'], ax=ax, cmap='inferno', fill=True, alpha=0.5, levels=50)
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
# plt.savefig("../images/03_mirage_att_heatmap.png")