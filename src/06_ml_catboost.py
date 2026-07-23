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
from catboost import CatBoostClassifier
 
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

# Tell CatBoost which predictors are categorical
cat_features = [
    X_train.columns.get_loc("wp")
]

base_model = CatBoostClassifier(
    loss_function="MultiClass",
    task_type="GPU",
    random_seed=42,
    verbose=0
)

param_distributions = {

    "iterations": [750],

    "learning_rate": loguniform(0.01, 0.2),

    "depth": randint(3, 10),

    "l2_leaf_reg": loguniform(1, 20),

    "bagging_temperature": uniform(0, 5),

    "random_strength": uniform(0, 5)

}

random_search = RandomizedSearchCV(

    estimator=base_model,

    param_distributions=param_distributions,

    n_iter=50,

    scoring="f1_macro",

    cv=5,

    random_state=42,

    n_jobs=1,

    verbose=3

)

random_search.fit(

    X_train,

    y_train,

    cat_features=cat_features,

    sample_weight=sample_weights

)

best_model = random_search.best_estimator_

predictions = best_model.predict(X_test)

print("\n--- Balanced Classification Report ---")

print(classification_report(y_test, predictions))

disp = ConfusionMatrixDisplay.from_estimator(

    best_model,

    X_test,

    y_test

)

disp.figure_.savefig("images/06_confusion_matrix_catboost.png")

print("Confusion matrix saved!")