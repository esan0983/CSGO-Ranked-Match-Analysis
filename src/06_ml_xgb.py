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
from xgboost import XGBRFClassifier
 
df = pd.read_csv("data/processed/csgo_cleaned_3.csv")
print("df info:")
print(df.info())
 
df2 = df[df["map"] == "de_mirage"]
df3 = df2[["total_dmg", "is_headshot", "wp", "inbetween_distance", "att_distance_to_bombsite", "vic_distance_to_bombsite", "att_tier", "vic_tier"]]
print("df3 info:")
print(df3.info())
 
custom_order = CategoricalDtype(categories=['Silver', 'Gold Nova', 'Master Guardian', 'Top Four'], ordered=True)
df3["att_tier"] = df3["att_tier"].astype(custom_order)
df3["vic_tier"] = df3["vic_tier"].astype(custom_order)
 
X1 = df3.drop(columns = ["att_tier", "vic_tier", "vic_distance_to_bombsite"])
X2 = df3.drop(columns = ["att_tier", "vic_tier", "att_distance_to_bombsite"])
y1 = df3["att_tier"].cat.codes
y2 = df3["vic_tier"].cat.codes
 
X1["is_headshot"] = X1["is_headshot"].astype(int)
X1["wp"] = X1["wp"].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X1,
    y1,
    test_size=0.2,
    random_state=42,
    stratify=y1
)

warnings.filterwarnings("ignore", category=UserWarning)

sample_weights = compute_sample_weight("balanced", y_train)

base_model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    device="cuda",
    enable_categorical=True,
    tree_method="hist",
    random_state=42
)

param_distributions = {

    "n_estimators":[750],

    "learning_rate":loguniform(0.01,0.2),

    "max_depth":randint(3,10),

    "subsample":uniform(0.6,0.4),

    "colsample_bytree":uniform(0.6,0.4),

    "min_child_weight":randint(1,10),

    "gamma":loguniform(1e-3,5),

    "reg_alpha":loguniform(1e-3,10),

    "reg_lambda":loguniform(1e-3,10)

}

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring="f1_macro",   # 'f1_macro' gives equal importance to all rank tiers
    cv=5,
    random_state=42,
    n_jobs=1,
    verbose=3
)

print(base_model.get_xgb_params())

random_search.fit(X_train, y_train, sample_weight=sample_weights)

best_model = random_search.best_estimator_
predictions = best_model.predict(X_test)

print("\n--- Balanced Classification Report ---")
print(classification_report(y_test, predictions))

disp = ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

disp.figure_.savefig('images/06_confusion_matrix_xgb.png')
print("Confusion matrix saved!")