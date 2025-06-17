import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, sep='|')
    print("Dataset loaded successfully. Columns:", df.columns.tolist())
    return df

def clean_data(df):
    df['TotalPremium'] = df['TotalPremium'].fillna(df['TotalPremium'].mean())
    df['TotalClaims'] = df['TotalClaims'].fillna(0)
    df['Gender'] = df['Gender'].fillna('Unknown')
    df['PostalCode'] = df['PostalCode'].fillna('Unknown')
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    Q1 = df['TotalClaims'].quantile(0.25)
    Q3 = df['TotalClaims'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['TotalClaims'] < (Q1 - 1.5 * IQR)) | (df['TotalClaims'] > (Q3 + 1.5 * IQR)))]
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    df = load_data('data/raw/MachineLearningRating_v3.txt')
    df_clean = clean_data(df)
    save_cleaned_data(df_clean, 'data/processed/insurance_data_cleaned.csv')