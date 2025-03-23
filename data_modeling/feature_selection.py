import pandas as pd
import numpy as np
from sklearn.calibration import LinearSVC
from sklearn.feature_selection import RFE

def perform_feature_selection(X, y, all_feature_names, top_k_all=150):
    print("\n--- Running RFE on all features ---")

    # Convert to DataFrame if necessary
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=all_feature_names)

    estimator = LinearSVC(max_iter=5000, dual=False)
    rfe = RFE(estimator, n_features_to_select=top_k_all)
    X_selected_array = rfe.fit_transform(X, y)
    selected_features = [all_feature_names[i] for i in rfe.get_support(indices=True)]

    print(f"\nFinal selected feature count: {len(selected_features)}")
    print("Selected features:", selected_features)

    return pd.DataFrame(X_selected_array, columns=selected_features), selected_features

# print("\n--- Running Chi-square on categorical features ---")
# cat_cols = [col for col in X.columns if col not in NUMERICAL_COLUMNS]
# chi_selector = SelectKBest(score_func=chi2, k=min(top_k_cat, len(cat_cols)))
# X_cat_selected_array = chi_selector.fit_transform(X[cat_cols], y)
# selected_cat_features = [cat_cols[i] for i in chi_selector.get_support(indices=True)]

# print("\n--- Running Mutual Info on categorical features ---")
# mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(top_k_cat, len(cat_cols)))
# X_cat_selected_array = mi_selector.fit_transform(X[cat_cols], y)
# selected_cat_features = [cat_cols[i] for i in mi_selector.get_support(indices=True)]

# print("\n--- Running RFE on numerical features ---")
# estimator = LinearSVC(max_iter=5000, dual=False)
# rfe = RFE(estimator, n_features_to_select=min(top_k_num, len(num_cols)))
# X_num_selected_array = rfe.fit_transform(X[num_cols], y)
# selected_num_features = [num_cols[i] for i in rfe.get_support(indices=True)]
