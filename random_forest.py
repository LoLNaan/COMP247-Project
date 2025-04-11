# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 03:21:23 2025

@author: Michael
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import os
import sys

# Add the root project folder to the path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from data_modeling.cleaning_pipeline import CleaningPipeline
#from data_modeling.transformation_pipeline import TransformationPipeline
#from data_modeling.feature_selection import perform_feature_selection

from cleaning_pipeline import CleaningPipeline
from transformation_pipeline import TransformationPipeline
from feature_selection import perform_feature_selection

# Load and clean the dataset
raw_df = pd.read_csv("KSI_dataset.csv")
clean_df = CleaningPipeline().transform(raw_df)

# Split target and features
y = clean_df['ACCLASS']
X = clean_df.drop(columns='ACCLASS')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=64)

# Apply the transformation pipeline
transformer = TransformationPipeline()
X_train_transformed = transformer.fit_transform(X_train, y_train)
X_test_transformed = transformer.transform(X_test)

# Feature Selection 
X_train_selected, selected_features = perform_feature_selection(X_train_transformed, y_train, transformer.get_feature_names())
X_test_selected = X_test_transformed[selected_features]

# Handle class imbalance in the training set with SMOTE (not on test set)
print("\nTraining values before SMOTE:\n", y_train.value_counts())
X_train_balanced, y_train_balanced = SMOTE(random_state=64).fit_resample(X_train_selected, y_train)
print("\nTraining values after SMOTE:\n", y_train_balanced.value_counts())

print(selected_features.dtypes)


######################
'''Building the model'''
rf_model = RandomForestClassifier(random_state=64, class_weight='balanced')

# Fit the initial Random Forest model on the balanced training data
rf_model.fit(X_train_balanced, y_train_balanced)

# Predict on the test set (original unbalanced)
y_pred = rf_model.predict(X_test_selected)

# Evaluate the model on the original test set
print("\nModel evaluation in the original testing set:")
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

##########################
'''Fine Tuning'''
parameters = {
    'n_estimators': range(50, 201, 50),        # Number of trees (between 50 and 200)
    'max_depth': [None, 10, 20, 30, 40],       # Maximum depth of trees
    'min_samples_split': [2, 5, 10],           # Minimum number of samples required to split an internal node 
    'min_samples_leaf': [1, 2, 4],             # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None],    # Number of features to consider when looking for the best split
    'bootstrap': [True, False],                # Whether bootstrap samples are used when building trees
    'criterion': ['gini', 'entropy']           # Function to measure the quality of a split
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_model,              # Random Forest model
                                   param_distributions=parameters,  # Hyperparameter grid
                                   n_iter=100,                      # Number of parameter settings to sample
                                   cv=5,                            # Number of folds for cross-validation
                                   n_jobs=-1,                       # Use all available cores
                                   verbose=1,                       # Verbosity level
                                   random_state=64)                 # Random state for reproducibility

# Fit the model using RandomizedSearchCV
random_search.fit(X_train_balanced, y_train_balanced)

# Best hyperparameters found by RandomizedSearchCV
print("Best parameters found:", random_search.best_params_)

# Evaluate the tuned model on the original test set
best_rf_random = random_search.best_estimator_
y_test_pred = best_rf_random.predict(X_test_selected)

print("\nTest accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Save the best estimator (fine-tuned model)
joblib.dump(best_rf_random, 'C:/Users/maiko/Downloads/RF_model.pkl')


































