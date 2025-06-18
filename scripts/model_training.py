import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Cleaned data loaded successfully. Columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def feature_engineering(df):
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Gender', 'PostalCode', 'Province', 'VehicleType', 'CoverType']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Extract date features
    if 'TransactionMonth' in df.columns:
        df['TransactionYear'] = pd.to_datetime(df['TransactionMonth'], errors='coerce').dt.year
        df['TransactionMonthNum'] = pd.to_datetime(df['TransactionMonth'], errors='coerce').dt.month
        df = df.drop('TransactionMonth', axis=1)
    
    # Drop irrelevant columns
    drop_cols = ['UnderwrittenCoverID', 'PolicyID'] if 'UnderwrittenCoverID' in df.columns else []
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    return df

def train_model(df, target_col='TotalClaims', model_path='models/rf_model.pkl', metrics_path='models/metrics.txt'):
    # Separate features and target
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        f.write(f"Mean Squared Error: {mse:.2f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}, metrics saved to {metrics_path}")
    return model, mse, r2

if __name__ == "__main__":
    df = load_data('data/processed/insurance_data_cleaned.csv')
    df = feature_engineering(df)
    model, mse, r2 = train_model(df)
    print(f"Training complete. MSE: {mse:.2f}, R2: {r2:.4f}")