# CSGO Ranked Analytics (WIP)

This project analyzes CSGO ranked match damage entry data to distinguish gameplay and behavior of players from different ranks. The workflow includes Pandas and SQL-based data cleaning, exploratory data analysis, feature engineering, statistical testing, predictive modeling, and a Tableau dashboard. The goal is to provide insight on what lower-ranked CSGO players do in comparison to higher-ranked players, thus potentially aiding in player training. Addtionally, with predictive modeling, this project hopes to make matchmaking more precise, especially for unranked players and smurf accounts.

## Motivation

There are two sources of motivation: helping players analyze their gameplay and making more accurate matchmaking guesses.

The concept of getting "hardstuck" in a rank of a competitive video game without knowing the reason is very common, especially because non-professionals don't have full access to game analysts, coaches, data, etc. This project attempts to bridge the gap using already-existing data to infer what a hardstuck player can do. Also, as mentioned, this project can serve as a potential solution to unranked matchmaking.

## Objectives

* Clean and validate raw damage entry data
* Explore patterns between player rank and a variety of data such as weapon type, round type, and positioning
* Engineer features for stronger statistical testing
* Predict player ranks from damage entry data
* Communicate insights through a dashboard

## Dataset

The dataset is from the following Kaggle link: https://www.kaggle.com/datasets/skihikingkevin/csgo-matchmaking-damage. The last file, "mm_master_demos.csv," has around 955k entries which spans over around 31500 rounds from 1400 ranked matches, recorded over the course of one month. It has 33 columns.

## Project Directory Structure

```text
|   README.md
+---dashboard
+---data (not included in repo)
|   \---processed  
|   \---raw         
+---images
+---notebooks
|   \---.ipynb_checkpoints           
+---reports
+---sql       
\---src
    \---__pycache__
```

## Project Workflow
```text
Raw Data
   ↓
Pandas Cleaning
   ↓
SQL Validation
   ↓
EDA
   ↓
Feature Engineering
   ↓
Statistical Analysis
   ↓
Machine Learning (WIP)
   ↓
Model Interpretation (WIP)
   ↓
Tableau Dashboard (WIP)
   ↓
Conclusion (WIP)
```

## Exploratory Data Analysis (EDA)

* For some weapons, such as the AK-47, there is a trend of proportional usage as rank increases.
* From the heatmaps, there are a lot of "confrontation hotspots," meaning that the heatmap of attacker coordinate data and victim coordinate data are nearly identical or just flipped in terms of frequency.
* The ranks of both attackers and victims approximately follow a normal distribution.
* There are proportionally more damage entries for eco rounds as rank goes up, and the inverse can be said for normal rounds.

## Feature Engineering 

* The in-between distance between the attacker and the victim was taken into account.
* The distance between the attacker/victim and the nearest bombsite is taken into account.
* Ranks (1-18) were binned into rank tiers: Silver, Gold Nova, Master Guardian, and Top Four
* "hitbox" data turned into an "is_headshot" bool
* Summed hp_dmg and arm_dmg to total_dmg

## Statistical Analysis 

* Chi-squared tests were used to confirm that the contingency pairs "att_tier/is_bomb_planted" and "att_tier/round_type" are related
* ANOVA and pairwise Tukey post-hoc analysis were used to analyze the trend of total damage per entry, inbetween distance, and distance to nearest bombsite as attacker rank tier goes up. All pairs had a statistically significant relationship for damage per entry and inbetween distance, while att_tier/att_distance_from_bombsite and vic_tier/vic_distance_from_bombsite had 5/6 pairs that had a statistically significant relationship, respectively.
* The Cochran-Armitage test was used to confirm that 14 weapons had a statistically significant trend as attacker rank tier goes up.

## Machine Learning (WIP)

For the first predictive model, I used XGBoost to predict attacker ranks. The inputs (and their justification) are as follows:
* total_dmg: higher ranked players can deal a lot more damage in one go due to spray control, recoil control, aim, etc.
* is_headshot: higher ranked players likely go for more headshots
* wp: some weapons' usage follow a trend as rank tier goes up (refer to the previous section)
* inbetween_distance: higher ranked players are more experienced shooting from farther distances
* att_distance_to_bombsite: higher ranked players are less likely to panic and rush in

Using a cross-validated random search with a reasonably large search space, the model achieved an accuracy of 0.49, which is almost twice as good as random guessing. The weighted f1 average (0.51) was higher than the macro f1 average (0.41), which indicates that the rarer tiers (Silver and Top Four) were much harder to predict, likely caused by significant class imbalance. Surprisingly, despite XGBoost not necessarily understanding the idea of ordinality of rank tiers, the misclassifications were usually of similar tiers, the most common being Gold Nova vs. Master Guardian. The model rarely made a Silver vs. Top Four mistake. This implies that the features were solid enough to display a general pattern. However, we can intuitively see from videos that there are stark differences between the gameplay of a rank 1 and a rank 18 player. Hence, we can safely assume that there's still room for more features, and said features were missed due to either a lack of feature engineering or a lack of game-relevant data from the dataset.

## Results (WIP)

## Dashboard (WIP)

## Technologies (WIP)

## Future Work (WIP)