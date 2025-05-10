"""
model_training.py

Pipeline for model training, cross-validation, and evaluation including scaling, SMOTE balancing,
and SHAP and permutation importance analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# === MODELS AND PARAM GRIDS ===
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2', 'l1'],
        'solver': ['liblinear'],
        'max_iter': [200, 500, 1000]
    },
    'Random Forest': {
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'XGBoost': {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'scale_pos_weight': [1, 2, 5]
    }
}

def run_and_save_fold_metrics(
    df,
    models,
    param_grids,
    cutoff_tag,
    feature_tag,
    output_csv,
    output_dir="saved_models",
    n_splits=10
):
    os.makedirs(output_dir, exist_ok=True)
    X = df.drop('Target', axis=1).values
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    results = []

    for model_name, model in models.items():
        print(f"Training {model_name} [{cutoff_tag}, {feature_tag}]")
        param_grid = param_grids[model_name].copy()

        if model_name == "XGBoost":
            counter = Counter(y_resampled)
            ratio = counter[0] / counter[1]
            param_grid['scale_pos_weight'] = [1, ratio, ratio * 2, 5]

        grid = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=-1)
        grid.fit(X_resampled, y_resampled)
        best_model = grid.best_estimator_

        joblib.dump(best_model, f"{output_dir}/{model_name.replace(' ', '_')}_{cutoff_tag}_{feature_tag}.pkl")
        joblib.dump(scaler, f"{output_dir}/scaler_{cutoff_tag}_{feature_tag}.pkl")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_test_scaled, y_test)):
            X_fold = X_test_scaled[test_idx]
            y_fold = y_test[test_idx]

            y_pred = best_model.predict(X_fold)
            y_proba = best_model.predict_proba(X_fold)[:, 1] if hasattr(best_model, "predict_proba") else None

            results.append({
                'cutoff': cutoff_tag,
                'features': feature_tag,
                'model': model_name,
                'fold': fold_idx + 1,
                'accuracy': accuracy_score(y_fold, y_pred),
                'precision': precision_score(y_fold, y_pred, average='weighted'),
                'recall': recall_score(y_fold, y_pred, average='weighted'),
                'f1': f1_score(y_fold, y_pred, average='weighted'),
                'auc': roc_auc_score(y_fold, y_proba) if y_proba is not None else np.nan
            })

        # SHAP Analysis (optional)
        try:
            print(f"Running SHAP for {model_name}...")
            feature_names = df.drop('Target', axis=1).columns
            X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
            shap_sample = min(100, len(X_test_df))
            X_shap = X_test_df.sample(shap_sample, random_state=42)

            if model_name in ["XGBoost", "Random Forest", "Gradient Boosting"]:
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_shap)
            else:
                explainer = shap.Explainer(best_model, X_shap)
                shap_values = explainer(X_shap)

            shap.summary_plot(shap_values, X_shap, show=False)
            plt.title(f"SHAP Summary - {model_name}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_summary_{model_name}_{cutoff_tag}_{feature_tag}.png")
            plt.close()
        except Exception as e:
            print(f"SHAP failed for {model_name}: {e}")

        # Permutation Importance
        try:
            print(f"Running permutation importance for {model_name}...")
            result = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1)
            importances = pd.DataFrame({
                'Feature': df.drop('Target', axis=1).columns,
                'Importance': result.importances_mean
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importances.head(10), palette='viridis')
            plt.title(f"Top 10 Permutation Importances - {model_name}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/perm_importance_{model_name}_{cutoff_tag}_{feature_tag}.png")
            plt.close()
        except Exception as e:
            print(f"Permutation importance failed for {model_name}: {e}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")
