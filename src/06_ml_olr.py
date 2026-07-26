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
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import ConfusionMatrixDisplay 
from statsmodels.miscmodels.ordinal_model import OrderedModel

df = pd.read_csv("data/processed/csgo_cleaned_3.csv")
print("df info:")
print(df.info())
 
df2 = df[df["map"] == "de_mirage"]
df3 = df2[["round_type", "is_bomb_planted", "total_dmg", "is_headshot", "wp", "inbetween_distance", "att_distance_to_bombsite", "vic_distance_to_bombsite", "att_tier", "vic_tier",
           "att_pos_x", "vic_pos_x"]]
print("df3 info:")
print(df3.info())
 
custom_order = CategoricalDtype(categories=['Silver', 'Gold Nova', 'Master Guardian', 'Top Four'], ordered=True)
df3["att_tier"] = df3["att_tier"].astype(custom_order)
df3["vic_tier"] = df3["vic_tier"].astype(custom_order)
 
X = df3.drop(columns=["att_tier", "vic_tier"])
y = df3["att_tier"].cat.codes

X = pd.get_dummies(X, columns=["wp", "round_type", "is_headshot", "is_bomb_planted"], drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

warnings.filterwarnings("ignore", category=UserWarning)

# Fit ordinal logistic regression
model = OrderedModel(
    y_train,
    X_train,
    distr="logit"
)

result = model.fit(method="bfgs")

# Predict probabilities
probabilities = result.model.predict(result.params, exog=X_test)

# Highest-probability class
predictions = probabilities.argmax(axis=1)

print(classification_report(y_test, predictions))

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions
)

disp.figure_.savefig("images/06_confusion_matrix_olr.png")