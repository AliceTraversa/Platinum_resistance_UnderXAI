#Part 1: Load the Dataset

import os
import pandas as pd

from data_loader import load_dataset

# Define the path to your Excel file
FILE_PATH = "data/your_dataset.xlsx"  # Update this with the correct path

# Load the dataset
df = load_dataset(FILE_PATH)

# Part 2: Dataset Filtering and Cleaning

from dataset_creation import create_filtered_dataset

# Apply initial filtering and cleaning
df = create_filtered_dataset(df)

#Part 3: Target Definition

from target_definition import define_target

# Define the binary classification target (e.g. 6 or 12 months cutoff)
cutoff_months = 6  # or 12 depending on the analysis
df = define_target(df, cutoff_months=cutoff_months)

#Part 4: Clinical Inspection & Encoding

from clinical_inspection import (
    remove_columns_with_too_many_nans,
    encode_combination_column,
    encode_surgical_interventions,
    finalize_dataset
)

# Step 1: Drop columns with too many NaNs
df = remove_columns_with_too_many_nans(df, threshold=400)

# Step 2: Encode multi-choice variables
# Replace below with actual column name lists from your dataset
chemotherapy_columns = [col for col in df.columns if col.startswith("Chemotherapy: drugs")]
comorbidity_columns = [col for col in df.columns if col.startswith("Comorbidities (choice=")]
cancer_columns = [col for col in df.columns if col.startswith("Type of previous/synchronous/metachronous cancer")]
surgery_open_columns = [col for col in df.columns if col.startswith("Type of Previous Open Abdominal Surgery")]
surgery_minimally_invasive_columns = [col for col in df.columns if col.startswith("Type of Previous Minimally Invasive Abdominal Surgery")]

df = encode_combination_column(df, chemotherapy_columns, 'Drugs')
df = encode_combination_column(df, comorbidity_columns, 'Comorbidities')
df = encode_combination_column(df, cancer_columns, 'Cancer')
df = encode_combination_column(df, surgery_open_columns, 'Surgery_Open')
df = encode_combination_column(df, surgery_minimally_invasive_columns, 'Surgery_Min_Invasive')

# Step 3: Encode surgical interventions
intervention_columns = [
    'Hysterectomy', 'Unilateral salpingo oophorectomy', 'Bilateral salpingo oophorectomy',
    'Colorectal resection WITH T-T anastomosis', 'Colorectal resection WITHOUT anastomosis',
    'Pelvic peritonectomy', 'Pelvic lymphadenectomy', 'Partial cystectomy', 'Ureteral resection',
    'Small bowel mesentery', 'Small bowel resection', 'Radical omentectomy',
    'Infracolic omentectomy', 'Appendectomy', 'Paraaortic lymphadenectomy', 'Large bowel resection',
    'Gutters\' peritonectomy', 'Diaphragmatic stripping', 'Diaphragmatic resection',
    'Morrison\'s pouch peritonectomy', 'Lesser omentum resection', 'Partial gastrectomy',
    'Celiac axis lymphadenectomy', 'Hepatic hilum lymphadenectomy', 'Splenectomy',
    'Partial pancreasectomy', 'Liver capsule resection', 'Atypical liver resection',
    'Partial hepatectomy', 'Cholecistectomy', 'Inguinal nodes', 'Pericardiophrenic nodes', 'Other'
]
df = encode_surgical_interventions(df, intervention_columns)

# Step 4: Select final dataset for modeling
df_selected = finalize_dataset(df)

#Part 5: Dataset Preprocessing

from preprocessing import clean_dataset

# Apply preprocessing: handle duplicates, NaNs, encode categoricals, compute features, etc.
df_cleaned = clean_dataset(df_selected)

#Part 6: Statistical Analysis

from stats_analysis import (
    convert_object_to_numeric,
    identify_variable_types,
    rename_columns,
    correlation_with_target,
    plot_correlation_matrix,
    plot_boxplots,
    plot_distribution,
    numerical_categorical_correlation,
    anova_test,
    plot_cross_correlation,
    remove_highly_correlated,
    remove_zero_variance
)

# Convert object columns to numeric where possible
df_numeric = convert_object_to_numeric(df_cleaned)

# Identify variable types
categorical_cols, continuous_cols = identify_variable_types(df_numeric)

# Optionally rename long/complex columns for clarity in plots (insert your renaming dict if needed)
# df_numeric = rename_columns(df_numeric, new_column_names)

# Correlation with target variable
correlation_with_target(df_numeric, target_col='Target')

# Plot correlation matrix
plot_correlation_matrix(df_numeric)

# Boxplots of continuous variables by target
plot_boxplots(df_numeric, target_col='Target', numeric_columns=continuous_cols)

# Distribution of categorical numeric variables
plot_distribution(df_numeric[categorical_cols])

# Correlation of categorical variables with target
numerical_categorical_correlation(df_numeric[categorical_cols], df_numeric['Target'])

# ANOVA test results for categorical variables
anova_df = anova_test(df_numeric[categorical_cols], df_numeric['Target'])
print(anova_df)

# Plot filtered correlation matrix to highlight multicollinearity
plot_cross_correlation(df_numeric[categorical_cols], threshold=0.5)

# Drop highly correlated or redundant columns if needed
df_numeric = remove_highly_correlated(df_numeric, column_to_drop='Surgical Complexity Score: cathegory   1 (low, <    4)  2 (intermediate, 4-7)  3 (high, >7)')

# Remove columns with zero variance
df_selected_clean = remove_zero_variance(df_numeric)

#Part 7: Feature Engineering

from feature_engineering import select_features_union_from_models

# Select important features using combined Logistic Regression and Random Forest
df_selected_features, selected_features = select_features_union_from_models(
    df_selected_clean,
    target_column='Target',
    n_features=15,
    random_state=42
)

print(f"\nFinal shape of dataset after feature selection: {df_selected_features.shape}")


#Part 8: Model Training and Evaluation

from model_training import run_and_save_fold_metrics, models, param_grids

# Define cutoff and feature tag
cutoff_tag = "6m"     # or "12m" depending on target definition
feature_tag = "reduced"  # or "full" if using the full set

# Output file path for metrics
output_csv = f"fold_metrics_{cutoff_tag}_{feature_tag}.csv"

# Run training, evaluation, SHAP, and permutation importance for all models
run_and_save_fold_metrics(
    df=df_selected_features,         # or df_selected_clean for full features
    models=models,
    param_grids=param_grids,
    cutoff_tag=cutoff_tag,
    feature_tag=feature_tag,
    output_csv=output_csv,
    output_dir="saved_models",       # optional: override if needed
    n_splits=10                      # 10-fold evaluation on the holdout test set
)

#Part 9: Statistical Results Analysis
from results_analysis import run_statistical_analysis

# Run statistical comparison tests
run_statistical_analysis()



