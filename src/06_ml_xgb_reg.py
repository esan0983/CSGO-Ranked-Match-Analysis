import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import os
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from scipy.stats import uniform, randint, loguniform
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb
from xgboost import XGBRFClassifier

 
df = pd.read_csv("data/processed/csgo_cleaned_3.csv")
print("df info:")
print(df.info())
 
df2 = df[df["map"] == "de_mirage"]
df3 = df2[["round_type", "is_bomb_planted", "total_dmg", "is_headshot", "wp", "inbetween_distance", "att_distance_to_bombsite", "vic_distance_to_bombsite", "att_rank",
           "att_pos_x", "vic_pos_x",]]
print(df3.info())

X = df3.drop(columns=["att_rank"])
y = df3["att_rank"]

X["is_headshot"] = X["is_headshot"].astype(int)
X["wp"] = X["wp"].astype("category")
X["round_type"] = X["round_type"].astype("category")
X["is_bomb_planted"] = X["is_bomb_planted"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

warnings.filterwarnings("ignore", category=UserWarning)

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

base_model = XGBRegressor(

    objective="reg:squarederror",

    device="cuda",

    tree_method="hist",

    enable_categorical=True,

    random_state=42

)

param_distributions = {

    "n_estimators": [750, 1000],

    "learning_rate": loguniform(0.01, 0.2),

    "max_depth": randint(3, 10),

    "subsample": uniform(0.6, 0.4),

    "colsample_bytree": uniform(0.6, 0.4),

    "min_child_weight": randint(1, 10),

    "gamma": loguniform(1e-3, 5),

    "reg_alpha": loguniform(1e-3, 10),

    "reg_lambda": loguniform(1e-3, 10)

}

random_search = RandomizedSearchCV(

    estimator=base_model,

    param_distributions=param_distributions,

    n_iter=50,

    scoring="neg_mean_absolute_error",

    cv=5,

    random_state=42,

    n_jobs=1,

    verbose=3

)

print(base_model.get_xgb_params())

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)

print("\n----- Regression Results -----")

print(f"MAE : {mae:.3f}")

print(f"RMSE: {rmse:.3f}")

print(f"R²  : {r2:.3f}")

print("Best Parameters:", random_search.best_params_)

# Convert predictions to the nearest valid rank
rounded_predictions = np.rint(predictions).astype(int)

# Keep predictions within the valid range of 1-18
rounded_predictions = np.clip(rounded_predictions, 1, 18)

# Absolute error in ranks
rank_errors = np.abs(rounded_predictions - y_test)

# Calculate percentages
exact = np.mean(rank_errors == 0)
within1 = np.mean(rank_errors <= 1)
within2 = np.mean(rank_errors <= 2)
within3 = np.mean(rank_errors <= 3)

print("\n----- Rank Accuracy -----")
print(f"Exact Rank     : {exact:.2%}")
print(f"Within 1 Rank  : {within1:.2%}")
print(f"Within 2 Ranks : {within2:.2%}")
print(f"Within 3 Ranks : {within3:.2%}")

print("Feature Importances:", best_model.feature_importances_)
xgb.plot_importance(best_model, importance_type='gain')