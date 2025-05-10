"""
target_definition.py

Functions for defining the classification target for platinum sensitivity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def define_target(df: pd.DataFrame, cutoff_months: int) -> pd.DataFrame:
    """
    Defines the binary target variable based on recurrence timing after platinum-based chemotherapy.

    If recurrence is later than the cutoff (e.g. >6 or >12 months), the patient is considered
    platinum-sensitive (target = 0). If recurrence is earlier or equal to the cutoff, they are 
    platinum-resistant (target = 1).

    Parameters:
        df (pd.DataFrame): Filtered dataset.
        cutoff_months (int): Cutoff in months to distinguish sensitivity from resistance (e.g. 6 or 12).

    Returns:
        pd.DataFrame: Dataset with a new 'target' column.
    """
    print(f"Applying target definition using a cutoff of {cutoff_months} months...")

    # Validation: recurrence should be NaN when there is no progression
    df_check = df[df['Progression/Recurrence after first line treatment?'] == 0]
    count_not_nan = df_check['Interval between last cycle of platinum-based chemotherapy and recurrence (months)'].notna().sum()
    if count_not_nan > 0:
        print(f"Warning: {count_not_nan} cases have non-NaN recurrence intervals despite no progression.")
    else:
        print("Validation passed: no unexpected values in recurrence interval for non-recurrence cases.")

    # Analyze recurrence distribution
    df_recurrence = df[df['Progression/Recurrence after first line treatment?'] == 1]
    count_nan = df_recurrence['Interval between last cycle of platinum-based chemotherapy and recurrence (months)'].isna().sum()
    print(f"{count_nan} NaN values found in recurrence interval for recurrence cases.")

    plt.figure(figsize=(10, 6))
    sns.histplot(
        df_recurrence['Interval between last cycle of platinum-based chemotherapy and recurrence (months)'].dropna(),
        bins=20, kde=True, color='skyblue', edgecolor='black'
    )
    plt.title(f'Distribution of recurrence interval (cutoff = {cutoff_months} months)', fontsize=14)
    plt.xlabel('Interval (months)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

    # Create binary target
    def classify(row):
        interval = row['Interval between last cycle of platinum-based chemotherapy and recurrence (months)']
        if pd.isna(interval) or interval > cutoff_months:
            return 0  # platinum-sensitive
        else:
            return 1  # platinum-resistant

    df['target'] = df.apply(classify, axis=1)

    print("\nTarget distribution (counts):")
    print(df['target'].value_counts())
    print("Target distribution (percentage):")
    print(df['target'].value_counts(normalize=True) * 100)

    return df
