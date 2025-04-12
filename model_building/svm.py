import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
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
raw_df = pd.read_csv("KSI_dataset.csv")
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

# Add data_modeling to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_MODELING_DIR = os.path.abspath(os.path.join(BASE_DIR, '../data_modeling'))
sys.path.append(DATA_MODELING_DIR)

# Save the transformation pipeline as a pkl file
joblib.dump(transformer, os.path.join(DATA_MODELING_DIR, 'transformation_pipeline.pkl'))
print("\nTransformation pipeline saved as 'transformation_pipeline.pkl'")

# Feature Selection using RFE with RandomForestClassifier
X_train_selected, selected_features = perform_feature_selection(X_train_transformed, y_train, transformer.get_feature_names())
X_test_selected = X_test_transformed[selected_features]

# Handle class imbalance in the training set with SMOTE
print("\nTraining values before SMOTE:\n", y_train.value_counts())
X_train_balanced, y_train_balanced = SMOTE(random_state=93).fit_resample(X_train_selected, y_train)
print("\nTraining values after SMOTE:\n", y_train_balanced.value_counts())

# Handle class imbalance in the testing set with SMOTE - this is useful to evaluate the model in a balanced scenario
print("\nTesting values before SMOTE:\n", y_test.value_counts())
X_test_balanced, y_test_balanced = SMOTE(random_state=93).fit_resample(X_test_selected, y_test)
print("\nTesting values after SMOTE:\n", y_test_balanced.value_counts())

# Train a SVM model
print("\n--- SVM model ---")

param_distributions = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf', 'poly']
}

random_search = RandomizedSearchCV(
    estimator=SVC(class_weight='balanced', random_state=93),
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
    verbose=2,
    n_jobs=-1
)

random_search.fit(X_train_balanced, y_train_balanced)
print("\nBest Params:", random_search.best_params_) # kernel='rbf', C=100, gamma=1

model = random_search.best_estimator_
model.fit(X_train_balanced, y_train_balanced)

# Evaluate the model - first with the original testing set, then with the balanced testing set
y_pred = model.predict(X_test_selected)

print("\nModel evaluation in the original testing set:")
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

y_pred_balanced = model.predict(X_test_balanced)

print("\n\nModel evaluation in the balanced testing set (after SMOTE):")
print("\nAccuracy:", round(accuracy_score(y_test_balanced, y_pred_balanced), 4))
print("\nClassification Report:\n", classification_report(y_test_balanced, y_pred_balanced))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_balanced, y_pred_balanced))

# Save the model as a pkl file
joblib.dump(model, 'svm_model.pkl')
print("\nModel saved as 'svm_model.pkl'")