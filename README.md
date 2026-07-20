# CSGO Ranked Analytics (WIP)

This project analyzes CSGO ranked match damage entry data to distinguish gameplay and behavior of players from different ranks. The workflow includes Pandas and SQL-based data cleaning, exploratory data analysis, feature engineering, statistical testing, predictive modeling, and a Tableau dashboard. The goal is to provide insight on what lower-ranked CSGO players do in comparison to higher-ranked players, thus potentially aiding in player training.

## Motivation

The concept of getting "hardstuck" in a rank of a competitive video game without knowing the reason is very common, especially because non-professionals don't have full access to game analysts, coaches, data, etc. This project attempts to bridge the gap using already-existing data to infer what a hardstuck player can do.

## Objectives

* Clean and validate raw damage entry data
* Explore patterns between player rank and a variety of data such as weapon type, round type, and positioning
* Engineer features for stronger statistical testing
* Predict player ranks from damage entry data
* Communicate insights through a dashboard

## Dataset

The dataset is from the following Kaggle link: https://www.kaggle.com/datasets/skihikingkevin/csgo-matchmaking-damage. The last file, "mm_master_demos.csv," has around 955k entries which spans over around 31500 rounds from 1400 ranked matches, recorded over the course of one month. It has 33 columns.

## Repository Structure

```text
|   README.md
|   
+---dashboard
+---data
|   +---processed
|   |       csgo_cleaned_1.csv
|   |       csgo_cleaned_2.csv
|   |       csgo_cleaned_3.csv
|   |       map_data.csv
|   |       
|   \---raw
|           map_data.csv
|           mm_master_demos.csv
|           
+---images
|       03_arm_dmg.png
|       03_att_rank_counts.png
|       03_avg_rank_counts.png
|       03_bomb_counts.png
|       03_bomb_site_counts.png
|       03_cache_vic_heatmap.png
|       03_cbble_vic_heatmap.png
|       03_dust2_vic_heatmap.png
|       03_hitbox_counts.png
|       03_hp_dmg.png
|       03_inferno_vic_heatmap.png
|       03_map_counts.png
|       03_mirage_vic_heatmap.png
|       03_overpass_vic_heatmap.png
|       03_rank_round_heatmap.png
|       03_rank_wp_heatmap.png
|       03_round_counts.png
|       03_round_type_counts.png
|       03_seconds.png
|       03_ticks.png
|       03_train_vic_heatmap.png
|       03_vic_rank_counts.png
|       03_wp_counts.png
|       03_wp_type_counts.png
|       05_jt_trend.png
|       de_cache.png
|       de_cbble.png
|       de_dust2.png
|       de_inferno.png
|       de_mirage.png
|       de_overpass.png
|       de_train.png
|       
+---notebooks
|   |   01_data_loading.ipynb
|   |   02_data_cleaning.ipynb
|   |   03_eda.ipynb
|   |   04_feature_engineering.ipynb
|   |   05_statistics.ipynb
|   |   
|   \---.ipynb_checkpoints
|           01_data_loading-checkpoint.ipynb
|           02_data_cleaning-checkpoint.ipynb
|           03_eda-checkpoint.ipynb
|           04_feature_engineering-checkpoint.ipynb
|           05_statistics-checkpoint.ipynb
|           
+---reports
+---sql
|       map_check.sql
|       
\---src
    |   02_data_cleaning.py
    |   jt_summary.py
    |   jt_test.py
    |   sqlcode.py
    |   
    \---__pycache__
            duckdb.cpython-314.pyc
            jt_summary.cpython-314.pyc
            jt_test.cpython-314.pyc
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

## Feature Engineering (WIP)

* The in-between distance between the attacker and the victim was taken into account.
* The distance between the attacker/victim and the nearest bombsite will be taken into account.

## Statistical Analysis

* Using the Cochran-Armitage test and the Benjamini-Hochberg procedure, we conclude that there is a statistically significant relationship between weapon used and attacker rank for 16 of the listed weapons.
* Using the Jonckheere-Terpstra test, we conclude that because of sample size, the extremely small asymptotic p-value might not necessarily mean there is a statistically significant relationship between attacker rank and the in-between distance between the attacker and the victim.

## Machine Learning (WIP)

## Results (WIP)

## Dashboard (WIP)

## Technologies (WIP)

## Future Work (WIP)