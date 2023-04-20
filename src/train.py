from process import *
from feature_extractor import *
import os
from os.path import join, pardir, curdir
import glob
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
from matplotlib import pyplot as plt
from joblib import load, dump
from sklearn.preprocessing import StandardScaler, label_binarize, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, classification_report
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Set current directory to "src"
os.chdir(join(os.getcwd(), os.pardir, "src"))
print(f"Current working directory: {os.getcwd()}")


DATASETS = ['tdcsfog', 'defog']
DATA_PATH = join(pardir, 'data')
PROCESSED_DATA_PATH = join(pardir, "data", "processed")
RANDOM_SEED = 42


X_train_df = load(join(PROCESSED_DATA_PATH, "X_train_df.joblib"))
X_train = load(join(PROCESSED_DATA_PATH, "X_train.joblib"))
y_train = load(join(PROCESSED_DATA_PATH, "y_train.joblib"))

y_train = y_train[~np.isnan(X_train).any(axis=1)]
X_train = X_train[~np.isnan(X_train).any(axis=1)]

X_val_df = load(join(PROCESSED_DATA_PATH, "X_val_df.joblib"))
X_val = load(join(PROCESSED_DATA_PATH, "X_val.joblib"))
y_val = load(join(PROCESSED_DATA_PATH, "y_val.joblib"))

y_val = y_val[~np.isnan(X_val).any(axis=1)]
X_val = X_val[~np.isnan(X_val).any(axis=1)]

if not os.path.isfile(join(pardir, "models", "scaler.joblib")):
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    dump(scaler, join(pardir, "models", "scaler.joblib"))
else:
    scaler = load(join(pardir, "models", "scaler.joblib"))
    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val)


if not os.path.isfile(join(pardir, "models", "selector_anova.joblib")):
    selector = SelectKBest(score_func=f_classif, k=40)
    selector = selector.fit(X_train_norm[:100000], y_train[:100000])
    X_train_selected = selector.transform(X_train_norm)
    X_val_selected = selector.transform(X_val_norm)
    dump(selector, join(pardir, "models", "selector_anova.joblib"))
else:
    selector = load(join(pardir, "models", "selector_anova.joblib"))
    X_train_selected = selector.transform(X_train_norm)
    X_val_selected = selector.transform(X_val_norm)

print(X_train_selected.shape, y_train.shape)

# Train XG Boost

model = XGBClassifier(verbosity=2)
model.fit(X_train_selected[:5000000], y_train[:5000000])

model.save_model(join(pardir, "models", "model_xgb_w_fs.json"))
# dump(model, join(pardir, "models", "model_xgb_w_fs.sav"))


print("Training Result")
y_pred = model.predict(X_train_selected)
print(classification_report(y_train, y_pred))

y_train_b = label_binarize(y_train, classes=[0, 1, 2, 3])
y_pred_b = label_binarize(y_pred, classes=[0, 1, 2, 3])
print(average_precision_score(y_train_b, y_pred_b))

print("Validation Result")
y_pred = model.predict(X_val_selected)
print(classification_report(y_val, y_pred))

y_val_b = label_binarize(y_val, classes=[0, 1, 2, 3])
y_pred_b = label_binarize(y_pred, classes=[0, 1, 2, 3])
print(average_precision_score(y_val_b, y_pred_b))
