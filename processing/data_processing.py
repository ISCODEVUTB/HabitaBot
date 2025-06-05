"""
data_processing.py

Purpose:
    This script contains functions to load, clean, and preview the Cartagena housing deficit dataset.
    It is designed as the first step in the HabitaBot data pipeline.

Main functions:
    - load_data: Reads the CSV file and returns a DataFrame, supporting semicolon delimiters and encoding issues.
    - clean_columns: Standardizes column names and data types.
    - check_nulls: Reports missing values in the dataset.
    - data_summary: Provides a basic statistical summary.
    - preview_data: Prints a preview of the first 10 rows.

Dependencies:
    - pandas
"""

import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and returns a DataFrame.
    Supports utf-8 and latin1/Windows encoded files. Assumes semicolon delimiter.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path, delimiter=";", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, delimiter=";", encoding="latin1")
    return df

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes column names and fixes data types.
    Args:
        df (pd.DataFrame): Original DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Keep column names as in original dataset (Spanish), but standardize format
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    cols_to_num = [
        "poblacion_total", "numero_hogares", "area_ha",
        "estrato_promedio", "deficit_habitacional"
    ]
    for col in cols_to_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if "año" in df.columns:
        df["año"] = pd.to_numeric(df["año"], downcast="integer", errors='coerce')
    return df

def check_nulls(df: pd.DataFrame):
    """
    Prints the count of null values per column.
    Args:
        df (pd.DataFrame): DataFrame to analyze.
    """
    print("\nNull values per column:")
    print(df.isnull().sum())

def data_summary(df: pd.DataFrame):
    """
    Prints a basic statistical summary of the dataset.
    Args:
        df (pd.DataFrame): DataFrame to analyze.
    """
    print("\nDescriptive summary:")
    print(df.describe(include='all'))

def preview_data(df: pd.DataFrame, n: int = 10):
    """
    Prints a preview of the first n rows of the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to preview.
        n (int): Number of rows to display.
    """
    # Display all columns as in the sample output
    with pd.option_context('display.max_columns', None, 'display.width', 1000):
        print(f"\nPreview of the first {n} rows:")
        print(df.head(n))

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'datos_sinteticos_cartagena_3200.csv')
    file_path = os.path.abspath(file_path)

    df = load_data(file_path)
    df = clean_columns(df)

    preview_data(df, n=10)
    check_nulls(df)
    data_summary(df)