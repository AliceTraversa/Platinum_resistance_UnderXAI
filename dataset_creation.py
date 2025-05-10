"""
dataset_creation.py

Functions for filtering and preparing the dataset for analysis.
"""

import pandas as pd

def create_filtered_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies initial filters and substitutions to prepare the dataset for analysis.

    Steps:
    - Removes rows where consent is missing
    - Converts 'Yes/No' and 'Checked/Unchecked' to 1/0
    - Removes rows with NaN in critical columns
    - Filters dataset based on clinical inclusion criteria

    Parameters:
        df (pd.DataFrame): Raw dataset loaded from Excel

    Returns:
        pd.DataFrame: Cleaned and filtered dataset
    """
    # Drop rows without patient consent
    df = df.dropna(subset=['Did the patient give their consent for data sharing?'])

    # Replace categorical values with binary
    df = df.replace({'Checked': 1, 'Unchecked': 0, 'Yes': 1, 'No': 0})

    # Remove rows with NaN in key treatment columns
    df = df.dropna(subset=['NACT', 'Did the patient undergo primary surgery?', 'Adjuvant/First line chemotherapy?'])

    # Remove 'Unknown' values
    df = df[df['Adjuvant/First line chemotherapy?'] != 'Unknown']

    # Apply clinical filters
    df = df[
        (df['NACT'] == 0) &
        (df['Did the patient undergo primary surgery?'] == 1) &
        (df['Adjuvant/First line chemotherapy?'] == 1)
    ]

    print(f"Dataset shape after filtering: {df.shape}")
    return df
