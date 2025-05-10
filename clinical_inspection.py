"""
clinical_inspection.py

Functions for clinical data inspection, transformation, encoding, and selection.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Remove columns that exceed a threshold of missing values (NaNs)
def remove_columns_with_too_many_nans(df: pd.DataFrame, threshold: int = 400) -> pd.DataFrame:
    # Identify columns with fewer or equal NaNs than the threshold
    columns_to_keep = df.columns[df.isnull().sum() <= threshold]
    # Filter dataframe to retain only those columns
    df = df[columns_to_keep]
    print(f"Dataset shape after removing columns with >{threshold} NaNs: {df.shape}")
    return df

# Encode binary multi-choice variables into single categorical codes
def encode_combination_column(df: pd.DataFrame, columns: list, new_column: str, exclude_label: str = 'No info') -> pd.DataFrame:
    # Extract sorted, human-readable labels from binary one-hot encoded columns
    def extract_combination(row):
        items = [col.split('(choice=')[-1].strip(')') for col, val in row.items() if val == 1]
        return ', '.join(sorted(items)) if items else exclude_label

    # Create combination label column
    df[new_column] = df[columns].apply(extract_combination, axis=1)
    # Count occurrences of each combination
    combinations = df[new_column].value_counts().reset_index()
    combinations.columns = ['Combination', 'Count']
    # Map each combination to a unique number
    combination_to_number = {comb: idx + 1 for idx, comb in enumerate(combinations['Combination'])}
    combination_to_number[exclude_label] = 0
    # Apply encoding to the column
    df[new_column] = df[new_column].map(combination_to_number).fillna(0).astype(int)
    # Drop original binary columns
    df.drop(columns=columns, inplace=True)
    return df

# Encode manually listed surgical intervention columns
def encode_surgical_interventions(df: pd.DataFrame, columns: list, new_column: str = 'Surgical Interventions') -> pd.DataFrame:
    # Create textual representation of interventions
    def extract_interventions(row):
        return ', '.join(sorted([col for col, val in row.items() if val == 1])) or 'No interventions'

    # Combine and encode intervention combinations
    df[new_column] = df[columns].apply(extract_interventions, axis=1)
    combination_counts = df[new_column].value_counts().reset_index()
    combination_counts.columns = ['Combination', 'Count']
    combination_to_number = {comb: idx + 1 for idx, comb in enumerate(combination_counts['Combination'])}
    combination_to_number['No interventions'] = 0
    df[new_column] = df[new_column].map(combination_to_number).fillna(0).astype(int)
    # Remove original columns
    df.drop(columns=columns, inplace=True)
    return df

# Select and return a subset of columns as final dataset for modeling or export
def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_select = [
        'Record ID', 'Previous/synchronous/metachronous cancer', 'Charlson Comorbidity Index',
        'Previous Open Abdominal Surgery', 'Previous Minimally Invasive Abdominal Surgery',
        'Height (cm) at the time of surgery', 'Weight (kg) at the time of surgery',
        'BMI (kg/m²) at the time of surgery', 'ASA Physical Status Class',
        'Eastern Cooperative Oncology Group (ECOG) Score', 'Ascites?',
        'Creatinine (mg/dL)', 'Albumin (g/dL)', 'Hemoglobin (g/dL)', 'Platelets (x1,000/mcl)',
        'Leukocytes (x1,000 U/mcl)  ', 'CA125 (U/mL)', 'HE4 (pmol/l)', 'Khorana score',
        'Length of Stay (days)', 'Age at Surgery', 'Surgical Approach', 'Surgical Complexity Score',
        'Surgical Complexity Score: cathegory   1 (low, <    4)  2 (intermediate, 4-7)  3 (high, >7)',
        'Residual Tumor (RT)', 'Anastomosis', 'Histotype', 'Epithelial carcinoma',
        'Peritoneal cytology', 'Pleural cytology',
        'Ovarian/Peritoneum/Fallopian Tube Cancer FIGO Staging  (based on clinical and pathological findings at the diagnosis)',
        'First cycle of CT: date', 'Last cycle of CT: date',
        'Total number of platinum-based cycles of adjuvant (post surgery) chemotherapy',
        'Drugs', 'Comorbidities', 'Cancer', 'Surgery_Open', 'Surgery_Min_Invasive',
        'Surgical Interventions', 'target'
    ]
    # Extract final set of relevant features
    df_selected = df[columns_to_select].copy()
    print("Final selected dataset preview:")
    print(df_selected.head())
    return df_selected
