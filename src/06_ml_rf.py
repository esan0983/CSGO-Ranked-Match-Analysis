import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)
import numpy as np
import warnings
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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
X["is_bomb_planted"] = X["is_bomb_planted"].astype(int)

X = pd.get_dummies(X, columns=["wp", "round_type"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

warnings.filterwarnings("ignore", category=UserWarning)

base_model = RandomForestRegressor(

    n_jobs=-1,

    bootstrap=True,

    random_state=42

)

param_distributions = {

    "n_estimators": [300],

    "max_depth": randint(3, 20),

    "min_samples_split": randint(10, 100),

    "min_samples_leaf": randint(5, 50),

    "max_features": uniform(0.3, 0.7),

    "max_samples": uniform(0.4, 0.4)

}

random_search = RandomizedSearchCV(

    estimator=base_model,

    param_distributions=param_distributions,

    n_iter=20,

    scoring="neg_mean_absolute_error",

    cv=5,

    random_state=42,

    n_jobs=1,

    verbose=3

)

print(base_model.get_params())

random_search.fit(
    X_train_sub,
    y_train_sub,
)

best_params = random_search.best_params_.copy()
best_params.pop("n_estimators", None)

step = 10
patience = 50
max_estimators = 2000

final_model = RandomForestRegressor(
    n_jobs=-1,
    bootstrap=True,
    warm_start=True,
    random_state=42,
    n_estimators=step,
    **best_params
)

best_val_mae = np.inf
best_n_estimators = step
no_improve_trees = 0

while True:
    final_model.fit(X_train_sub, y_train_sub)

    val_preds = final_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_n_estimators = final_model.n_estimators
        no_improve_trees = 0
    else:
        no_improve_trees += step

    if no_improve_trees >= patience or final_model.n_estimators >= max_estimators:
        break

    final_model.n_estimators += step

final_model.n_estimators = best_n_estimators
final_model.estimators_ = final_model.estimators_[:best_n_estimators]

print(f"\nStopped at {best_n_estimators} trees (best val MAE: {best_val_mae:.3f})")

best_model = final_model

predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)

print("\n----- Regression Results -----")

print(f"MAE : {mae:.3f}")

print(f"RMSE: {rmse:.3f}")

print(f"R²  : {r2:.3f}")

print("Best Parameters:", random_search.best_params_)

rounded_predictions = np.rint(predictions).astype(int)

rounded_predictions = np.clip(rounded_predictions, 1, 18)

rank_errors = np.abs(rounded_predictions - y_test)

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