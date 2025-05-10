"""
feature_engineering.py

Feature selection using a combination of Logistic Regression with L1 penalty and Random Forest importance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def select_features_union_from_models(df_selected_clean, target_column='Target', n_features=15, random_state=42):
    print("Starting feature selection using Logistic Regression (L1) and Random Forest\n")

    # Split data into features (X) and target (y)
    X = df_selected_clean.drop(columns=target_column)
    y = df_selected_clean[target_column]
    feature_names = X.columns.tolist()

    # --- 1. Logistic Regression with L1 penalty ---
    log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, random_state=random_state)
    log_reg.fit(X, y)
    coefs = np.abs(log_reg.coef_[0])
    top_logreg_idx = np.argsort(coefs)[-n_features:]
    top_logreg_features = [feature_names[i] for i in top_logreg_idx if coefs[i] > 0]

    print(f"Top {len(top_logreg_features)} features from Logistic Regression (L1):")
    for i, feat in enumerate(top_logreg_features, 1):
        print(f"  {i:2d}. {feat}")

    # --- 2. Random Forest Importance ---
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X, y)
    importances = rf.feature_importances_
    top_rf_idx = np.argsort(importances)[-n_features:]
    top_rf_features = [feature_names[i] for i in top_rf_idx]

    print(f"\nTop {len(top_rf_features)} features from Random Forest:")
    for i, feat in enumerate(top_rf_features, 1):
        print(f"  {i:2d}. {feat}")

    # --- 3. Union of features ---
    selected_features_union = sorted(set(top_logreg_features + top_rf_features))
    print(f"\nFinal selected features (union): {len(selected_features_union)}")
    for i, feat in enumerate(selected_features_union, 1):
        print(f"  {i:2d}. {feat}")

    # --- 4. Return reduced dataset ---
    df_reduced = X[selected_features_union].copy()
    df_reduced[target_column] = y.values
    return df_reduced, selected_features_union
