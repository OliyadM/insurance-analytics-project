import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

# --- Sklearn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# --- Model Imports ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def load_data(file_path):
    """Loads the cleaned dataset."""
    print("Loading cleaned data...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully from {file_path}. Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def preprocess_for_severity_modeling(df):
    """
    Prepares the data for the claim severity model.
    - Filters for policies with claims.
    - Dynamically selects available features.
    - Creates a preprocessing pipeline.
    """
    print("Preprocessing data for severity modeling...")
    
    # 1. --- Filtering for the modeling goal ---
    # We predict TotalClaims for policies that HAVE a claim.
    df_claimed = df[df['TotalClaims'] > 0].copy()
    
    if len(df_claimed) == 0:
        raise ValueError("No data with claims > 0 found. Cannot build severity model.")
        
    # 2. --- DYNAMIC Feature Selection (This is the fix) ---
    TARGET = 'TotalClaims'
    
    # Define a list of desired features we'd like to use if they exist
    DESIRED_NUMERIC_FEATURES = ['CustomValueEstimate', 'Kilowatts', 'Cubiccapacity', 'VehicleAge']
    DESIRED_CATEGORICAL_FEATURES = ['Province', 'Gender', 'VehicleType', 'Make', 'Bodytype', 'MaritalStatus']
    
    # Check which of the desired features are actually in the dataframe
    available_numeric_features = [f for f in DESIRED_NUMERIC_FEATURES if f in df_claimed.columns]
    available_categorical_features = [f for f in DESIRED_CATEGORICAL_FEATURES if f in df_claimed.columns]
    
    # Print a clear message about which features were found and which were missing
    all_desired = set(DESIRED_NUMERIC_FEATURES + DESIRED_CATEGORICAL_FEATURES)
    all_available = set(available_numeric_features + available_categorical_features)
    missing_features = list(all_desired - all_available)
    
    print(f"Found {len(all_available)} out of {len(all_desired)} desired features.")
    if missing_features:
        print(f"WARNING: The following features were not found and will be skipped: {missing_features}")

    FEATURES = available_numeric_features + available_categorical_features
    X = df_claimed[FEATURES]
    y = df_claimed[TARGET]

    # 3. --- Create Preprocessing Pipelines ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. --- Combine pipelines with ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, available_numeric_features),
            ('cat', categorical_transformer, available_categorical_features)
        ],
        remainder='passthrough'
    )
    
    print("Preprocessing setup complete.")
    return X, y, preprocessor

def train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    Trains and evaluates regression models. Returns results and the best model.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, objective='reg:squarederror')
    }
    
    results = {}
    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'RMSE': rmse, 'R-squared': r2}
        print(f"Results for {name}: RMSE = {rmse:.2f}, R-squared = {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_pipeline
            print(f"*** New best model: {name} (R-squared: {r2:.4f}) ***")
            
    return results, best_model

def analyze_feature_importance_with_shap(model_pipeline, X_test, output_dir):
    """Uses SHAP to analyze and plot feature importances for the best model."""
    print("\n--- Analyzing Feature Importance with SHAP ---")
    
    preprocessor = model_pipeline.named_steps['preprocessor']
    model = model_pipeline.named_steps['regressor']
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    try:
        cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(X_test.select_dtypes(include='object').columns)
    except:
        cat_feature_names = [] # Handle case with no categorical features
    
    feature_names = list(X_test.select_dtypes(include=np.number).columns) + list(cat_feature_names)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    explainer = shap.Explainer(model, X_test_transformed_df)
    shap_values = explainer(X_test_transformed_df)

    # --- Generate and Save SHAP Plots ---
    # Use plt.figure() to create a new figure for each plot to prevent overlap
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_bar.png'))
    plt.close()
    print("Saved SHAP summary bar plot.")

    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed_df, show=False)
    plt.title("SHAP Feature Importance (Beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_beeswarm.png'))
    plt.close()
    print("Saved SHAP beeswarm plot.")

if __name__ == "__main__":
    DATA_PATH = 'data/processed/insurance_data_cleaned.csv'
    MODEL_DIR = 'models'
    PLOT_DIR = 'plots'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    df = load_data(DATA_PATH)
    
    if df is not None:
        X, y, preprocessor = preprocess_for_severity_modeling(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")

        results, best_model = train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor)
        
        print("\n--- Model Comparison ---")
        results_df = pd.DataFrame(results).T
        print(results_df.sort_values(by='R-squared', ascending=False))
        
        if best_model:
            model_path = os.path.join(MODEL_DIR, 'best_severity_model.joblib')
            joblib.dump(best_model, model_path)
            print(f"\n✅ Best performing severity model saved successfully to: {model_path}")
            
            analyze_feature_importance_with_shap(best_model, X_test, PLOT_DIR)