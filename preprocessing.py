"""
preprocessing.py

Functions for preprocessing and transforming the dataset prior to modeling.
"""

import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate rows if any
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        print(f"Removing {n_duplicates} duplicate rows")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found. All good!")

    # Display NaN counts
    nan_counts = df.isna().sum()
    print("NaN counts per column:")
    print(nan_counts)

    # Drop rows with any NaN or 'Unknown' values
    df.replace("Unknown", pd.NA, inplace=True)
    df = df.dropna()
    print(f"Shape after dropping NaNs and 'Unknown': {df.shape}")

    # Drop specific columns not needed for modeling
    df = df.drop(columns=[
        'Previous/synchronous/metachronous cancer',
        'Previous Open Abdominal Surgery',
        'Previous Minimally Invasive Abdominal Surgery',
        'ASA Physical Status Class',
        'Record ID'
    ], errors='ignore')

    # Create new feature: treatment interval (in months)
    df['First cycle of CT: date'] = pd.to_datetime(df['First cycle of CT: date'])
    df['Last cycle of CT: date'] = pd.to_datetime(df['Last cycle of CT: date'])
    df['treatment interval'] = df.apply(
        lambda row: (row['Last cycle of CT: date'] - row['First cycle of CT: date']).days / 30
        if pd.notna(row['First cycle of CT: date']) and pd.notna(row['Last cycle of CT: date'])
        else np.nan,
        axis=1
    )
    df.drop(columns=['First cycle of CT: date', 'Last cycle of CT: date'], inplace=True)

    # Map categorical values to integers for several columns
    df['Pleural cytology'] = df['Pleural cytology'].replace({
        'Not Done/No Effusion': 0,
        'Negative': 1,
        'Positive': 2
    }).astype(int)

    df['Peritoneal cytology'] = df['Peritoneal cytology'].replace({
        'Not Done/No Ascites': 0,
        'Negative': 1,
        'Sample not suitable for diagnosis': 2,
        'Positive': 3
    }).astype(int)

    df['Eastern Cooperative Oncology Group (ECOG) Score'] = df['Eastern Cooperative Oncology Group (ECOG) Score'].replace({
        '0 - Fully active, able to carry on all pre-disease performance without restriction': 0,
        '1 - Restricted in physically strenuous activity but ambulatory and able to carry out work of a light or sedentary nature, e.g., light house work, office work': 1,
        '2 - Ambulatory and capable of all selfcare but unable to carry out any work activities. Up and about more than 50% of waking hours': 2
    }).astype(int)

    # One-hot encode FIGO staging and epithelial carcinoma
    df = pd.get_dummies(df, columns=[
        'Ovarian/Peritoneum/Fallopian Tube Cancer FIGO Staging  (based on clinical and pathological findings at the diagnosis)',
        'Epithelial carcinoma'
    ], dtype=int, drop_first=False)

    print(f"Final shape after preprocessing: {df.shape}")
    return df
