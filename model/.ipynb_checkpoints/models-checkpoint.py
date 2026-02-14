#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    roc_auc_score
)

# Preprocess
def preprocess_data(X, y):

    # Encode target if categorical
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Encode features
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test


# Train models
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train, y_train)
    return model


def train_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def train_randomforest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model


# Evaluate
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob[:, 1])
    except:
        auc = "Not applicable"

    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "AUC": auc,
        "Matthews Correlation Coefficient (MCC Score)": mcc
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return metrics, cm, report

