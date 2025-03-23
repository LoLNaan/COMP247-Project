import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import os
import sys

# Add the root project folder to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_modeling.cleaning_pipeline import CleaningPipeline
from data_modeling.transformation_pipeline import TransformationPipeline
# from data_modeling.feature_selection import perform_feature_selection

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

# Feature Selection - This step is too aggressive and reduces the model's performance so it's commented out
# X_train_selected, selected_features = perform_feature_selection(X_train_transformed, y_train)
# X_test_selected = X_test_transformed[selected_features]

# Handle class imbalance in the training set with SMOTE
print("\nTraining values before SMOTE:\n", y_train.value_counts())
# X_train_balanced, y_train_balanced = SMOTE(random_state=93).fit_resample(X_train_selected, y_train)
X_train_balanced, y_train_balanced = SMOTE(random_state=93).fit_resample(X_train_transformed, y_train)
print("\nTraining values after SMOTE:\n", y_train_balanced.value_counts())

# Train a Simple SVM
print("\n--- Training the SVM model ---")
model = SVC(kernel='rbf', C=10, gamma=0.01, class_weight='balanced', random_state=93)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate the model
# y_pred = model.predict(X_test_selected)
y_pred = model.predict(X_test_transformed)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
