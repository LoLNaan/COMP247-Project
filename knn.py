# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:41:31 2025

@author: Jerin Gogi
"""
import os
import sys
#Getting root directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
from Data_Modeling.cleaning_pipeline import CleaningPipeline
from Data_Modeling.transformation_pipeline import TransformationPipeline



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, KFold
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
#Applying SMOTE and resampling training data
X_train, y_train = SMOTE(random_state = 27).fit_resample(transformed_train, y_train)

print ("\n======Model before finetuing======\n")
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
y_pred = knn.predict(transformed_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n",classification_report(y_test, y_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))



print ("\n======Finetuing======\n")

# Define hyperparameters for finetuning 
kf = KFold(n_splits = 5, shuffle = True, random_state = 27)
param_grid = {'n_neighbors': np.arange(2,30,1),
              'weights':['distance','uniform']
              }

#Creating a KNN instance 
knn = KNeighborsClassifier()

#Using grid search to fine and using the hyperameters
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid,
                           cv=kf, scoring='accuracy', verbose=1)

# Fit on the SMOTE-balanced training set
grid_search.fit(X_train, y_train)

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
y_pred = best_knn.predict(transformed_test)

# Calculating Accuracy
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report: \n",classification_report(y_test, y_pred))
print("Confusion matrix \n",confusion_matrix(y_test,y_pred))

joblib.dump(best_knn, r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\Model\KNN_model.pkl')





