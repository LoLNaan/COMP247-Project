import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import os
import sys

# Add the root project folder to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_modeling.cleaning_pipeline import CleaningPipeline
from data_modeling.transformation_pipeline import TransformationPipeline
from data_modeling.feature_selection import perform_feature_selection

# Load and clean the dataset
raw_df = pd.read_csv("../KSI_dataset.csv")
clean_df = CleaningPipeline().transform(raw_df)

# Split target and features
y = clean_df['ACCLASS']
X = clean_df.drop(columns='ACCLASS')

# Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=93)

# Apply the transformation pipeline
transformer = TransformationPipeline()
X_train_transformed = transformer.fit_transform(X_train_raw, y_train)
X_test_transformed = transformer.transform(X_test_raw)

# Feature Selection using RFE with RandomForestClassifier
X_train_selected, selected_features = perform_feature_selection(X_train_transformed, y_train, transformer.get_feature_names())
X_test_selected = X_test_transformed[selected_features]

# Handle class imbalance in the training set with SMOTE
print("\nTraining values before SMOTE:\n", y_train.value_counts())
X_train_balanced, y_train_balanced = SMOTE(random_state=93).fit_resample(X_train_selected, y_train)
print("\nTraining values after SMOTE:\n", y_train_balanced.value_counts())

# Handle class imbalance in the testing set with SMOTE
print("\nTesting values before SMOTE:\n", y_test.value_counts())
X_test_balanced, y_test_balanced = SMOTE(random_state=93).fit_resample(X_test_selected, y_test)
print("\nTesting values after SMOTE:\n", y_test_balanced.value_counts())

# Train a logistic regression model
iterations = [100, 200, 500, 1000, 2000, 5000]
c = [0.001, 0.01, 0.1, 1, 10, 100]
tol = [1e-3] # reduced tolerance to make program run faster, and manual testing showed higher precision did not improve results
param_grid = [
    {   # L1 Regularization (works with only liblinear or saga)
        'penalty': ['l1'],
        'solver': ['liblinear', 'saga'],
        'C': c,
        'max_iter': iterations,
        'tol': tol,
    },
    {   # L2 Regularization (works with most solvers)
        'penalty': ['l2'],
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'C': c,
        'max_iter': iterations,
        'tol': tol,
    },
    {   # ElasticNet (only works with saga)
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.1, 0.5, 0.9],  # Mix of L1/L2
        'C': c,
        'max_iter': iterations,
        'tol': tol,
    },
    {   # No regularization (None)
        'penalty': [None],
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'max_iter': iterations,
        'tol': tol,
    }
]

# Train with GridSearchCV
model = GridSearchCV(
    LogisticRegression(random_state=57,
                       class_weight='balanced',
                       # tol=1e-2,
                       ),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
model.fit(X_train_balanced, y_train_balanced)

print(f"\nBest parameters found: \n{model.best_params_}")

# Evaluate the model - first with the original testing set, then with the balanced testing set
y_pred = model.predict(X_test_selected)

print(f"\nModel evaluation in the original testing set:")
print(f"\nAccuracy: {round(accuracy_score(y_test, y_pred), 4) * 100}%", )
print(f"\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

y_pred_balanced = model.predict(X_test_balanced)

print(f"\n\nModel evaluation in the balanced testing set (after SMOTE):")
print(f"\nAccuracy: {round(accuracy_score(y_test_balanced, y_pred_balanced), 4) * 100}%")
print(f"\nClassification Report:\n{classification_report(y_test_balanced, y_pred_balanced)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test_balanced, y_pred_balanced)}")

# Save the model as a pkl file
joblib.dump(model, 'logreg_model.pkl')
print("\nModel saved as 'logreg_model.pkl'")