import pandas as pd

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the ovarian cancer dataset from an Excel file.

    Parameters:
        filepath (str): Path to the Excel file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_excel(filepath)
    print(f"Initial dataset shape: {df.shape}")
    return df
