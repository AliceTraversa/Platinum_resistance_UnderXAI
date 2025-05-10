"""
stats_analysis.py

Statistical analysis and visualization functions for exploring variable distributions and correlations.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

def convert_object_to_numeric(df):
    # Convert object columns to numeric if possible
    object_columns = df.select_dtypes(include=['object']).columns
    print("Object columns before conversion:", object_columns.tolist())
    for col in object_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def identify_variable_types(df):
    # Separate numerical into categorical vs continuous
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = [col for col in numeric_columns if df[col].nunique() < 10]
    continuous_columns = [col for col in numeric_columns if df[col].nunique() >= 10]
    print("Categorical columns:", categorical_columns)
    print("Continuous columns:", continuous_columns)
    return categorical_columns, continuous_columns

def rename_columns(df, new_column_names):
    df = df.rename(columns=new_column_names)
    print("Renamed columns preview:")
    print(df.head())
    return df

def correlation_with_target(df, target_col='Target'):
    # Correlation with target
    df_no_target = df.drop(columns=[target_col])
    corr = df_no_target.apply(lambda col: col.corr(df[target_col]))
    print("Correlation with target:")
    print(corr)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=corr.index, y=corr.values, palette='coolwarm')
    plt.xticks(rotation=90)
    plt.title("Correlation with Target")
    plt.tight_layout()
    plt.show()
    return corr

def plot_correlation_matrix(df):
    # Full correlation heatmap
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

def plot_boxplots(df, target_col, numeric_columns):
    # Boxplot for each numeric column vs target
    numeric_columns = [col for col in numeric_columns if col != target_col]
    n_vars = len(numeric_columns)
    n_cols = 4
    n_rows = int(np.ceil(n_vars / n_cols))
    plt.figure(figsize=(16, 5 * n_rows))
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(x=target_col, y=col, data=df, palette="coolwarm")
        plt.title(f"Boxplot of {col} by {target_col}")
    plt.tight_layout()
    plt.show()

def plot_distribution(df):
    # Plot distribution for each numerical categorical variable
    num_vars = len(df.columns)
    rows = (num_vars // 3) + (num_vars % 3 > 0)
    plt.figure(figsize=(15, 5 * rows))
    for i, col in enumerate(df.columns):
        plt.subplot(rows, 3, i + 1)
        sns.histplot(df[col], bins=20, kde=True, color="skyblue")
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

def numerical_categorical_correlation(df, target):
    corr = df.corrwith(target).sort_values(ascending=False)
    plt.figure(figsize=(20, 12))
    sns.barplot(x=corr.values, y=corr.index, palette="coolwarm")
    plt.title("Pearson Correlation with Target")
    plt.xlabel("Correlation")
    plt.show()
    return corr

def anova_test(df, target):
    results = {}
    unique_vals = target.unique()
    for col in df.columns:
        groups = [df[col][target == val] for val in unique_vals]
        f_stat, p_val = f_oneway(*groups)
        results[col] = p_val
    return pd.DataFrame.from_dict(results, orient='index', columns=['p-value']).sort_values(by='p-value')

def plot_cross_correlation(df, threshold=0.3):
    corr_matrix = df.corr()
    mask = np.abs(corr_matrix) < threshold
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, mask=mask)
    plt.title(f"Filtered Correlation Matrix (threshold={threshold})")
    plt.show()

def remove_highly_correlated(df, column_to_drop):
    return df.drop(columns=[column_to_drop], errors='ignore')

def remove_zero_variance(df):
    std_devs = df.std()
    zero_std_cols = std_devs[std_devs == 0].index.tolist()
    print("Columns with zero variance:", zero_std_cols)
    return df.drop(columns=zero_std_cols)
