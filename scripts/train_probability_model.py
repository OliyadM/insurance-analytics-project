import pandas as pd
import numpy as np
import os
import joblib

# --- Sklearn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# --- Model Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_data(file_path):
    """Loads the cleaned dataset from the specified path."""
    print("Loading cleaned data...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully from {file_path}. Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Make sure you have run the data preprocessing script.")
        return None

def preprocess_for_classification(df):
    """
    Prepares the data for the claim probability model.
    - Dynamically selects available features.
    - Defines the binary target 'HasClaim'.
    - Creates a robust preprocessing pipeline.
    """
    print("Preprocessing data for classification modeling...")
    
    # 1. --- Target Definition ---
    # We will use the 'HasClaim' column which was created in your data_preprocessing.py script.
    TARGET = 'HasClaim'
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Please ensure it's created in your preprocessing script.")

    # 2. --- DYNAMIC Feature Selection (This is the fix to prevent errors) ---
    # Define a list of desired features we want to use if they exist.
    DESIRED_NUMERIC_FEATURES = ['CustomValueEstimate', 'Kilowatts', 'Cubiccapacity', 'VehicleAge']
    DESIRED_CATEGORICAL_FEATURES = ['Province', 'Gender', 'VehicleType', 'Make', 'Bodytype', 'MaritalStatus']
    
    # Check which of the desired features are actually in the dataframe
    available_numeric_features = [f for f in DESIRED_NUMERIC_FEATURES if f in df.columns]
    available_categorical_features = [f for f in DESIRED_CATEGORICAL_FEATURES if f in df.columns]
    
    # Print a clear message about which features were found and which were missing
    all_desired = set(DESIRED_NUMERIC_FEATURES + DESIRED_CATEGORICAL_FEATURES)
    all_available = set(available_numeric_features + available_categorical_features)
    missing_features = list(all_desired - all_available)
    
    print(f"Found {len(all_available)} out of {len(all_desired)} desired features for modeling.")
    if missing_features:
        print(f"WARNING: The following features were not found and will be skipped: {missing_features}")

    FEATURES = available_numeric_features + available_categorical_features
    X = df[FEATURES]
    y = df[TARGET]

    # 3. --- Create Preprocessing Pipelines ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, available_numeric_features),
            ('cat', categorical_transformer, available_categorical_features)
        ])
    
    print("Preprocessing setup complete.")
    return X, y, preprocessor

def train_and_evaluate_classifiers(X_train, y_train, X_test, y_test, preprocessor):
    """
    Trains and evaluates classification models. Returns results and the best model.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    best_model = None
    best_auc = 0.0

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                                         ('classifier', model)])
        model_pipeline.fit(X_train, y_train)
        
        y_pred = model_pipeline.predict(X_test)
        y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {'Accuracy': accuracy, 'AUC': auc}
        print(f"Results for {name}: Accuracy = {accuracy:.4f}, AUC = {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        if auc > best_auc:
            best_auc = auc
            best_model = model_pipeline
            print(f"*** New best model: {name} (AUC: {auc:.4f}) ***")
            
    return results, best_model

if __name__ == "__main__":
    DATA_PATH = 'data/processed/insurance_data_cleaned.csv'
    MODEL_DIR = 'models'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = load_data(DATA_PATH)
    
    if df is not None:
        X, y, preprocessor = preprocess_for_classification(df)
        
        # stratify=y is critical for imbalanced classification to ensure train/test sets have similar class distributions.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")

        results, best_classifier = train_and_evaluate_classifiers(X_train, y_train, X_test, y_test, preprocessor)
        
        print("\n--- Final Classifier Comparison ---")
        results_df = pd.DataFrame(results).T
        print(results_df.sort_values(by='AUC', ascending=False))
        
        if best_classifier:
            model_path = os.path.join(MODEL_DIR, 'best_probability_model.joblib')
            joblib.dump(best_classifier, model_path)
            print(f"\n✅ Best performing probability model saved successfully to: {model_path}")