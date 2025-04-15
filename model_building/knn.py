# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:41:31 2025

@author: Jerin Gogi
"""
import os
import sys
#Getting root directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
from data_modeling.cleaning_pipeline import CleaningPipeline
from data_modeling.transformation_pipeline import TransformationPipeline
from data_modeling.feature_selection import perform_feature_selection


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, KFold
import joblib
#Load the dataset
data = pd.read_csv(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\KSI_dataset.csv')

cleaner = CleaningPipeline() #Create CleaningPipeline instance
transformer = TransformationPipeline() #Create Transformation instance

data2 = cleaner.transform(data)# Cleaning the data

#Defining the features and target columns
X = data2.drop(columns = 'ACCLASS')
Y = data2['ACCLASS']

#Splitting training and testing data
X_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, stratify  = Y, random_state = 93)

#Trasforming train data
transformed_train = transformer.fit_transform(X_train,y_train)
#Transforming test data
transformed_test = transformer.transform(x_test)
print(transformed_test.columns.tolist())

#Performing feature selection

X_train_selected, selected_features = perform_feature_selection(transformed_train, y_train, transformer.get_feature_names())
X_test_selected = transformed_test[selected_features]
print(selected_features)

#Applying SMOTE and resampling training data
print("\nTraining values before SMOTE:\n", y_train.value_counts())
X_train_balanced, y_train_balanced = SMOTE(random_state=93).fit_resample(X_train_selected, y_train)
print("\nTraining values after SMOTE:\n", y_train_balanced.value_counts())
print(X_train_balanced.columns.tolist())
# Handle class imbalance in the testing set with SMOTE - this is useful to evaluate the model in a balanced scenario
print("\nTesting values before SMOTE:\n", y_test.value_counts())
X_test_balanced, y_test_balanced = SMOTE(random_state=93).fit_resample(X_test_selected, y_test)
print("\nTesting values after SMOTE:\n", y_test_balanced.value_counts())

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






