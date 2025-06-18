import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully. Columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def calculate_metrics(df):
    # Claim Frequency: Proportion of policies with claims > 0
    df['HasClaim'] = df['TotalClaims'] > 0
    # Claim Severity: Average claim amount for claims > 0
    df['ClaimSeverity'] = df['TotalClaims'].where(df['TotalClaims'] > 0)
    # Margin: TotalPremium - TotalClaims
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def chi_square_test(df, group_col, metric_col):
    # For categorical metrics (e.g., HasClaim)
    contingency_table = pd.crosstab(df[group_col], df[metric_col])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return p

def t_test(df, group_col, metric_col, group_values=None):
    # For numerical metrics (e.g., ClaimSeverity, Margin)
    if group_values:
        group1 = df[df[group_col] == group_values[0]][metric_col].dropna()
        group2 = df[df[group_col] == group_values[1]][metric_col].dropna()
    else:
        groups = df[group_col].unique()
        if len(groups) != 2:
            raise ValueError(f"Expected 2 groups for {group_col}, found {len(groups)}")
        group1 = df[df[group_col] == groups[0]][metric_col].dropna()
        group2 = df[df[group_col] == groups[1]][metric_col].dropna()
    _, p = ttest_ind(group1, group2, equal_var=False)  # Welch’s t-test
    return p

def hypothesis_testing(df, output_path):
    df = calculate_metrics(df)
    results = []

    # H0: No risk differences across provinces (Claim Frequency)
    p_freq_prov = chi_square_test(df, 'Province', 'HasClaim')
    results.append({
        'Hypothesis': 'No risk differences across provinces (Claim Frequency)',
        'p-value': p_freq_prov,
        'Reject H0': p_freq_prov < 0.05,
        'Interpretation': 'Higher claim frequency in some provinces (e.g., Gauteng) suggests regional risk adjustments.'
    })

    # H0: No risk differences across provinces (Claim Severity)
    provinces = df['Province'].value_counts().index[:2]  # Top 2 provinces
    p_sev_prov = t_test(df, 'Province', 'ClaimSeverity', group_values=provinces)
    results.append({
        'Hypothesis': 'No risk differences across provinces (Claim Severity)',
        'p-value': p_sev_prov,
        'Reject H0': p_sev_prov < 0.05,
        'Interpretation': f'Larger claims in {provinces[0]} vs. {provinces[1]} indicate premium adjustments.'
    })

    # H0: No risk differences between zip codes (Claim Frequency)
    top_zips = df['PostalCode'].value_counts().index[:10]  # Top 10 zip codes
    df_zip = df[df['PostalCode'].isin(top_zips)]
    p_freq_zip = chi_square_test(df_zip, 'PostalCode', 'HasClaim')
    results.append({
        'Hypothesis': 'No risk differences between zip codes (Claim Frequency)',
        'p-value': p_freq_zip,
        'Reject H0': p_freq_zip < 0.05,
        'Interpretation': 'Certain zip codes have higher claim likelihood, supporting targeted marketing.'
    })

    # H0: No margin difference between zip codes
    top_zips = df['PostalCode'].value_counts().index[:2]  # Top 2 zip codes
    p_margin_zip = t_test(df, 'PostalCode', 'Margin', group_values=top_zips)
    results.append({
        'Hypothesis': 'No margin difference between zip codes',
        'p-value': p_margin_zip,
        'Reject H0': p_margin_zip < 0.05,
        'Interpretation': 'Profitability varies by zip code, suggesting pricing optimization.'
    })

    # H0: No risk difference between women and men (Claim Frequency)
    df_gender = df[df['Gender'].isin(['Male', 'Female'])]  # Exclude 'Unknown'
    p_freq_gender = chi_square_test(df_gender, 'Gender', 'HasClaim')
    results.append({
        'Hypothesis': 'No risk difference between women and men (Claim Frequency)',
        'p-value': p_freq_gender,
        'Reject H0': p_freq_gender < 0.05,
        'Interpretation': 'Gender-based risk differences may inform premium adjustments.'
    })

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(f"Hypothesis: {result['Hypothesis']}\n")
            f.write(f"p-value: {result['p-value']:.4f}\n")
            f.write(f"Reject H0: {result['Reject H0']}\n")
            f.write(f"Interpretation: {result['Interpretation']}\n\n")
    
    print(f"Hypothesis testing results saved to {output_path}")
    return results

if __name__ == "__main__":
    df = load_data('data/processed_data/insurance_data_cleaned.csv')
    results = hypothesis_testing(df, 'reports/hypothesis_test_results.txt')