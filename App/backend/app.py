from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import sys
import joblib
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Add the root project folder to the path
sys.path.append(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project')

# Load the transformation pipeline and models
transformation_pipeline = joblib.load(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\data_modeling\transformation_pipeline.pkl') 
svm_model = joblib.load(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\Model\svm_model.pkl')
logreg_model = joblib.load(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\Model\logreg_model.pkl')
knn_model = joblib.load(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\Temp\KNN2_model.pkl')
rf_model = joblib.load(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\Model\rf_model.pkl')
neural_network_model = joblib.load(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\Model\neural_network_model.pkl')

print("All models and pipeline loaded successfully.")

# The client only provides the selected features, but the transformation pipeline works on the entire dataset.
# Therefore, we need to add default values the missing columns to the input data before applying the transformation pipeline.
def add_missing_columns_for_transformation(input_data):
    required_columns = {'DISABILITY', 'CYCLIST', 'ALCOHOL', 'HOOD_158', 'DRIVACT', 'INJURY', 'MOTORCYCLE', 'EMERG_VEH'}
    missing_columns = required_columns - set(input_data.columns)

    for col in missing_columns:
        input_data[col] = 'No'

    return input_data

# This function aligns the transformed data with the model's expected features.
# This is necessary because we use a custom feature selection when fitting the models.
def align_features(transformed_data, transformed_feature_names, expected_features):
    # Create a DataFrame from the transformed data and its feature names
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_feature_names)

    # Keep only the columns that match the expected features
    aligned_df = transformed_df[expected_features]

    # Return the aligned data as a numpy array
    return aligned_df.to_numpy()

# Prediction route
@app.route('/predict', methods=['POST'])
def predict_log():
    try:
        # Get JSON data from the request
        data = request.get_json()        

        input_df = pd.DataFrame([{
            'TIME': int(data.get('TIME', 0)),
            'ROAD_CLASS': str(data.get('ROAD_CLASS', '')),
            'DISTRICT': str(data.get('DISTRICT', '')),
            'LATITUDE': float(data.get('LATITUDE', 0)),
            'LONGITUDE': float(data.get('LONGITUDE', 0)),
            'ACCLOC': str(data.get('ACCLOC', '')),
            'TRAFFCTL': str(data.get('TRAFFCTL', '')),
            'VISIBILITY': str(data.get('VISIBILITY', '')),
            'LIGHT': str(data.get('LIGHT', '')),
            'RDSFCOND': str(data.get('RDSFCOND', '')),
            'IMPACTYPE': str(data.get('IMPACTYPE', '')),
            'INVTYPE': str(data.get('INVTYPE', '')),
            'INVAGE': str(data.get('INVAGE', '')), 
            'INJURY': str(data.get('INJURY', '')), 
            'INITDIR': str(data.get('INITDIR', '')), 
            'VEHTYPE': str(data.get('VEHTYPE', '')), 
            'MANOEUVER': str(data.get('MANOEUVER', '')),
            'DRIVACT': str(data.get('DRIVACT', '')),
            'DRIVCOND': str(data.get('DRIVCOND', '')),
            'DAY': int(data.get('DAY', 1)),
            'MONTH': int(data.get('MONTH', 1)),
            'WEEKDAY': str(data.get('WEEKDAY', '')),
            'PEDESTRIAN': "Yes" if data.get('PEDESTRIAN', False) else "No",
            'CYCLIST': "Yes" if data.get('CYCLIST', False) else "No",
            'AUTOMOBILE': "Yes" if data.get('AUTOMOBILE', False) else "No",
            'MOTORCYCLE': "Yes" if data.get('MOTORCYCLE', False) else "No",
            'TRUCK': "Yes" if data.get('TRUCK', False) else "No",
            'TRSN_CITY_VEH': "Yes" if data.get('TRSN_CITY_VEH', False) else "No",
            'EMERG_VEH': "Yes" if data.get('EMERG_VEH', False) else "No",
            'PASSENGER': "Yes" if data.get('PASSENGER', False) else "No",
            'SPEEDING': "Yes" if data.get('SPEEDING', False) else "No",
            'AG_DRIV': "Yes" if data.get('AG_DRIV', False) else "No",
            'REDLIGHT': "Yes" if data.get('REDLIGHT', False) else "No",
            'ALCOHOL': "Yes" if data.get('ALCOHOL', False) else "No",
            'DISABILITY': "Yes" if data.get('DISABILITY', False) else "No",
            'HOOD_158': str(data.get('HOOD_158', ''))
        }])

        pd.set_option('display.max_columns', None)

        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth',None)
        print(input_df)

        input_df = add_missing_columns_for_transformation(input_df)
        # Transform the input data
        transformed_data = transformation_pipeline.transform(input_df)

        # Select model
        model_name = data.get('model', 'logreg').lower()
        metrics = {}
        if model_name == 'svm':
            model = svm_model
            metrics = {
                'accuracy': 0.53,
                'precision': 0.75,
                'recall': 0.53,
                'f1_score': 0.41
            }
        if model_name == 'logreg':
            model = logreg_model
            metrics = {
                "accuracy": 0.77,
                "precision": 0.78,
                "recall": 0.77,
                "f1_score": 0.77
            }
        elif model_name == 'neural_network':
           model = neural_network_model
           metrics = {
                "accuracy": 0.88,
                "precision": 0.89,
                "recall": 0.88,
                "f1_score": 0.88
            }
        elif model_name == 'rf':
            model = rf_model
            metrics = {
                "accuracy": 0.94,
                "precision": 0.95,
                "recall": 0.94,
                "f1_score": 0.94
            }
        elif model_name == 'knn':
            model = knn_model
            metrics = {
                "accuracy": 0.88,
                "precision": 0.89,
                "recall": 0.88,
                "f1_score": 0.88
            }
        else:
            return jsonify({ "error": "Invalid model selected" }), 400

        # Align the transformed data with the model's expected features
        transformed_data_aligned = align_features(
            transformed_data, 
            transformation_pipeline.get_feature_names(), 
            model.feature_names_in_
        )

        # Make predictions
        prediction = model.predict(transformed_data_aligned)
        prediction_proba = model.predict_proba(transformed_data_aligned)
        return jsonify({
            'prediction': str(prediction[0]),
            'confidence': float(max(prediction_proba[0])),
            'metrics': metrics,

        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(debug=True)


    