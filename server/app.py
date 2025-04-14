from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import sys
import joblib

app = Flask(__name__)

# Add the root project folder to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the transformation pipeline and models
transformation_pipeline = joblib.load('../data_modeling/transformation_pipeline.pkl')  # Adjusted to relative path from root
svm_model = joblib.load('../model_building/svm_model.pkl')
logreg_model = joblib.load('../model_building/logreg_model.pkl')
# knn_model = joblib.load('../model_building/knn_model.pkl')
rf_model = joblib.load('../model_building/rf_model.pkl')
neural_network_model = joblib.load('../model_building/neural_network_model.pkl')

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
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert to DataFrame and add missing columns
        input_df = pd.DataFrame([data])
        input_df = add_missing_columns_for_transformation(input_df)

        # Transform the input data
        transformed_data = transformation_pipeline.transform(input_df)

        # Select model
        model_name = data.get('model', 'svm').lower()
        if model_name == 'svm':
            model = svm_model
        elif model_name == 'logreg':
            model = logreg_model
        elif model_name == 'neural_network':
            model = neural_network_model
        elif model_name == 'rf':
            model = rf_model
        # elif model_name == 'knn':
        #     model = knn_model
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

        return jsonify({ "prediction": prediction.tolist() })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(debug=True)