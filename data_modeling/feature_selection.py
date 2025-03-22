import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression

'''
This function is meant feature selection using chi-square for categorical features and RFE for numerical features.
I tried using this approach on top of the cleaned and transformed dataset, but it seems to be too aggressive.
It removes too many features which reduces the model's performance significantly.
'''
def perform_feature_selection(X: pd.DataFrame, y: pd.Series, top_k_cat=20, top_k_num=5):
    print("\n--- Running feature selection ---")

    # Separate categorical and numerical columns
    # Explicitly declare the known numeric columns
    num_cols = ['TIME', 'LATITUDE', 'LONGITUDE']
    # Everything else is treated as categorical
    cat_cols = [col for col in X.columns if col not in num_cols]

    # One-hot encode categorical columns for chi-square
    X_cat_encoded = pd.get_dummies(X[cat_cols], drop_first=True)
    X_cat_encoded = X_cat_encoded.astype(np.float64)  # Ensure numeric type

    # Remove any columns with negative values for chi-square
    X_cat_encoded = X_cat_encoded.loc[:, (X_cat_encoded >= 0).all()]

    print(f"\nChi2 will use {X_cat_encoded.shape[1]} one-hot encoded features")

    chi_selector = SelectKBest(score_func=chi2, k=min(top_k_cat, X_cat_encoded.shape[1]))
    chi_selector.fit(X_cat_encoded, y)
    chi_selected = X_cat_encoded.columns[chi_selector.get_support()].tolist()

    # RFE on numerical features
    print("\nRunning RFE on numerical columns...")
    X_num = X[num_cols]
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=min(top_k_num, X_num.shape[1]))
    rfe.fit(X_num, y)
    rfe_selected = X_num.columns[rfe.support_].tolist()

    selected_features = chi_selected + rfe_selected

    print(f"\nFinal selected feature count: {len(selected_features)}")
    print(f"\nSelected features: {selected_features}")

    return X[selected_features].copy(), selected_features
