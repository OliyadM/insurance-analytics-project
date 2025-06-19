import pandas as pd
import numpy as np
import csv

def load_data(file_path):
    """Load the raw dataset."""
    try:
        df = pd.read_csv(file_path, sep='|', encoding='latin1', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        print("Dataset loaded successfully. Columns:", df.columns.tolist())
        return df
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def clean_data(df):
    """Clean the dataset and perform feature engineering."""
    # Fill missing values
    df['TotalPremium'] = df['TotalPremium'].fillna(df['TotalPremium'].mean())
    df['TotalClaims'] = df['TotalClaims'].fillna(0)
    df['Gender'] = df['Gender'].fillna('Unknown')
    df['PostalCode'] = df['PostalCode'].fillna('Unknown')
    df['Province'] = df['Province'].fillna('Unknown')
    df['VehicleType'] = df['VehicleType'].fillna('Unknown')
    
    # Clean mixed-type columns
    if 'CapitalOutstanding' in df.columns:
        df['CapitalOutstanding'] = pd.to_numeric(df['CapitalOutstanding'], errors='coerce')
        df['CapitalOutstanding'] = df['CapitalOutstanding'].fillna(df['CapitalOutstanding'].median())
    if 'CrossBorder' in df.columns:
        df['CrossBorder'] = df['CrossBorder'].astype(str).str.strip().fillna('Unknown')
    
    # Convert datetime
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    # Feature engineering
    if 'RegistrationYear' in df.columns:
        df['VehicleAge'] = 2024 - df['RegistrationYear'].astype(float)
    df['HasClaim'] = df['TotalClaims'] > 0
    df['ClaimSeverity'] = df['TotalClaims'].where(df['HasClaim'])
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
    
    # Remove outliers for non-zero claims
    if df['TotalClaims'].sum() > 0:
        Q1 = df['TotalClaims'][df['TotalClaims'] > 0].quantile(0.25)
        Q3 = df['TotalClaims'][df['TotalClaims'] > 0].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df['TotalClaims'] == 0) | (
            (df['TotalClaims'] >= (Q1 - 1.5 * IQR)) & 
            (df['TotalClaims'] <= (Q3 + 1.5 * IQR))
        )
        df = df[mask]
        print(f"Removed outliers; new dataset size: {len(df)} rows.")
    
    return df

def save_cleaned_data(df, output_path):
    """Save the cleaned dataset."""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    df = load_data('data/raw/MachineLearningRating_v3.txt')
    df_clean = clean_data(df)
    save_cleaned_data(df_clean, 'data/processed/insurance_data_cleaned.csv')