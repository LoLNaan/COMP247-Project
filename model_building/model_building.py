from matplotlib import pyplot as plt
import pandas as pd
import joblib
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
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




'''
Model 1: SVM
'''

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
    n_jobs=-1,
    random_state=93,
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



'''
Model 2: Logistic Regression
'''

print("\n--- Logistic Regression model ---")

from sklearn.linear_model import LogisticRegression

# Setup model parameters
iterations = list(range(500, 6000, 500)) # 500 to 5500 in steps of 500
c = [0.001, 0.01, 0.1, 1, 10, 100]
tol = 1e-3

param_grid = [
    {   # L1 Regularization (works with only liblinear or saga)
        'penalty': ['l1'],
        'solver': ['liblinear', 'saga'],
        'C': c,
        'max_iter': iterations,
    },
    {   # L2 Regularization (works with most solvers)
        'penalty': ['l2'],
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'C': c,
        'max_iter': iterations,
    },
    {   # ElasticNet (only works with saga)
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.1, 0.5, 0.9],  # Mix of L1/L2
        'C': c,
        'max_iter': iterations,
    },
    {   # No regularization (None)
        'penalty': [None],
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'max_iter': iterations,
    }
]

# Train with RandomizedSearchCV
model = RandomizedSearchCV(
    estimator=LogisticRegression(random_state=57,
                                 class_weight='balanced',
                                 tol=1e-3,
                                 # reduced tolerance to make program run faster.
                                 # manual testing with GridSearchCV showed lower tol did not improve results, often exceeding max_iter,
                                 # and higher tol would cut off too early.
                                 ),

    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=57
)
model.fit(X_train_balanced, y_train_balanced)
print(f"\nBest parameters found: \n{model.best_params_}")

# Evaluate the model - first with the original testing set, then with the balanced testing set
y_pred = model.predict(X_test_selected)

print(f"\nModel evaluation in the original testing set:")
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%", )
print(f"\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

y_pred_balanced = model.predict(X_test_balanced)

print(f"\n\nModel evaluation in the balanced testing set (after SMOTE):")
print(f"\nAccuracy: {accuracy_score(y_test_balanced, y_pred_balanced) * 100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test_balanced, y_pred_balanced)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test_balanced, y_pred_balanced)}")

# Save the model as a pkl file
joblib.dump(model, 'logreg_model.pkl')
print("\nModel saved as 'logreg_model.pkl'")



'''
Model 3: Neural Network
'''

print("\n--- Neural Network model ---")
# Train a "Neural Network" model
from sklearn.neural_network import MLPClassifier

# Train with MLP Classifier
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
    activation='relu',             # ReLU activation function
    solver='adam',                 # Adam optimizer
    alpha=0.0001,                  # L2 regularization parameter
    batch_size='auto',             # Batch size for gradient descent
    learning_rate='adaptive',      # Adaptive learning rate
    max_iter=300,                  # Maximum number of iterations
    random_state=93                # Random state for reproducibility
)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate the model - first with the original testing set, then with the balanced testing set
y_pred = model.predict(X_test_selected)

print(f"\nModel evaluation in the original testing set:")
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%", )
print(f"\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

y_pred_balanced = model.predict(X_test_balanced)

print(f"\n\nModel evaluation in the balanced testing set (after SMOTE):")
print(f"\nAccuracy: {accuracy_score(y_test_balanced, y_pred_balanced) * 100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test_balanced, y_pred_balanced)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test_balanced, y_pred_balanced)}")

# Save the model as a pkl file
joblib.dump(model, 'neural_network_model.pkl')
print("\nModel saved as 'neural_network_model.pkl'")



'''
Model 4: Random Forest
'''

print("\n--- Random Forest model ---")

from sklearn.ensemble import RandomForestClassifier

######################
'''Building the model'''
rf_model = RandomForestClassifier(random_state=93, class_weight='balanced')

# Fit the initial Random Forest model on the balanced training data
rf_model.fit(X_train_balanced, y_train_balanced)

# Predict on the test set (original unbalanced)
y_pred = rf_model.predict(X_test_selected)

# Evaluate the model on the original test set
print("\nModel evaluation in the original testing set:")
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

y_pred_balanced = rf_model.predict(X_test_balanced)

print("\n\nModel evaluation in the balanced testing set (after SMOTE):")
print("\nAccuracy:", round(accuracy_score(y_test_balanced, y_pred_balanced), 4))
print("\nClassification Report:\n", classification_report(y_test_balanced, y_pred_balanced))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_balanced, y_pred_balanced))


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
                                   random_state=93)                 # Random state for reproducibility

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
joblib.dump(best_rf_random, 'rf_model.pkl')



'''
Model 5: KNN
'''

print("\n--- KNN model ---")
#Creating a KNN instance 
knn = KNeighborsClassifier()
print ("\n======Model before finetuing======\n")
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_balanced, y_train_balanced)
y_pred = knn.predict(X_test_selected)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n",classification_report(y_test, y_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))



print ("\n======Finetuing======\n")

# Define hyperparameters for finetuning 
kf = KFold(n_splits = 5, shuffle = True, random_state = 93)
param_grid = {'n_neighbors': np.arange(2,30,1),
              'weights':['distance','uniform']
              }
#Using grid search to fine and using the hyperameters
grid_search = RandomizedSearchCV(estimator=knn,
                                 param_distributions=param_grid,
                                 cv=kf, 
                                 scoring='accuracy',
                                 n_iter=20,
                                 verbose=2,
                                 n_jobs=-1)

# Fit on the SMOTE-balanced training set
grid_search.fit(X_train_balanced, y_train_balanced)

#Plotting the accuracy for different number of neighbors for each weight type 
results = grid_search.cv_results_
df_results = pd.DataFrame(results)

plt.figure(figsize=(10, 6))

for weight in param_grid['weights']:
    # Filter rows where weight matches
    mask = df_results['param_weights'] == weight
    k_values = df_results[mask]['param_n_neighbors']
    scores = df_results[mask]['mean_test_score']
    
    plt.plot(k_values, scores, label=f'weights = {weight}', marker='o')

plt.title('KNN Accuracy vs Number of Neighbors for Each Weight Type')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print ("\n======Model after finetuing======\n")
# Best estimator and score
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Use best model to predict
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_selected)

# Calculating Accuracy
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report: \n",classification_report(y_test, y_pred))
print("Confusion matrix \n",confusion_matrix(y_test,y_pred))

joblib.dump(best_knn, r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\Temp\KNN2_model.pkl')

