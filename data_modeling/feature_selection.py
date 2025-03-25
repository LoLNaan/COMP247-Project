import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFE

def perform_feature_selection(X: pd.DataFrame, y: pd.Series, all_feature_names: list):
    print("\n--- Running Feature Selection using Random Forest Importances ---")
    # Ensure X is a DataFrame with correct column names
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=all_feature_names)

    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_

    # Set threshold as mean importance
    threshold = np.mean(importances)

    # Select features above threshold
    selected_indices = np.where(importances > threshold)[0]
    selected_features = [all_feature_names[i] for i in selected_indices]

    print(f"Selected {len(selected_features)} features out of {len(all_feature_names)}")
    print("Top selected features (by importance):")
    for name, score in sorted(zip(selected_features, importances[selected_indices]), key=lambda x: -x[1]):
        print(f"{name}: {score:.4f}")

    # Return reduced dataset and feature names
    return X[selected_features], selected_features

# def perform_feature_selection(X, y, all_feature_names, top_k_all=150):
#     print("\n--- Running RFE with RandomForestClassifier ---")

#     # Convert to DataFrame if necessary
#     if not isinstance(X, pd.DataFrame):
#         X = pd.DataFrame(X, columns=all_feature_names)

#     # Use a non-linear estimator (tree-based)
#     estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     rfe = RFE(estimator, n_features_to_select=top_k_all, step=10)
    
#     X_selected_array = rfe.fit_transform(X, y)
#     selected_features = [all_feature_names[i] for i in rfe.get_support(indices=True)]

#     print(f"\nFinal selected feature count: {len(selected_features)}")
#     print("Selected features:", selected_features)

#     return pd.DataFrame(X_selected_array, columns=selected_features), selected_features
