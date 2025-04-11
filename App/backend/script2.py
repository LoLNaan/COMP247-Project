from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback
import numpy as np
import sys 
import os
sys.path.append(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project')
from Data_Modeling.transformation_pipeline import TransformationPipeline
from Data_Modeling.cleaning_pipeline import CleaningPipeline
from flask_cors import CORS

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
CORS(app)

# Global variables for model and transformer
knn_model = None
transformer = None
feature_names = None

def initialize():
    global knn_model, transformer, feature_names
    
    try:
        # Load model
        knn_model = joblib.load(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\Model\KNN_model.pkl')
        print("Model loaded successfully")
        
        # Load training data to get feature structure
        train_data = pd.read_csv(r'C:\Users\Jerin Gogi\Downloads\COMP 247\Project\KSI_dataset.csv')
        cleaner = CleaningPipeline()
        cleaned_data = cleaner.transform(train_data)
        X_train = cleaned_data.drop(columns='ACCLASS')
        
        # Initialize and fit transformer
        transformer = TransformationPipeline()
        transformed_data = transformer.fit_transform(X_train)
        feature_names = transformer.get_feature_names()
        print("Transformer initialized with feature names:", feature_names)

        
        
    except Exception as e:
        print("Initialization error:", str(e))
        print(traceback.format_exc())
        raise

# Initialize when starting
initialize()

@app.route('/api/models/knn/predict', methods=['POST'])
def predict():
    try:
        if not knn_model or not transformer:
            return jsonify({'error': 'Model not initialized'}), 500

        # Get data from frontend
        data = request.json
        
        # Create input DataFrame with same structure as training data
        input_data = pd.DataFrame([{
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
            'MANOEUVER': str(data.get('MANOEUVER', '')),
            'DRIVACT': str(data.get('DRIVACT', '')),
            'DRIVCOND': str(data.get('DRIVCOND', '')),
            'DAY': int(data.get('DAY', 1)),
            'MONTH': int(data.get('MONTH', 1)),
            'WEEKDAY': int(data.get('WEEKDAY', 0)),
            'PEDESTRIAN': int(data.get('PEDESTRIAN', False)),
            'CYCLIST': int(data.get('CYCLIST', False)),
            'AUTOMOBILE': int(data.get('AUTOMOBILE', False)),
            'MOTORCYCLE': int(data.get('MOTORCYCLE', False)),
            'TRUCK': int(data.get('TRUCK', False)),
            'TRSN_CITY_VEH': int(data.get('TRSN_CITY_VEH', False)),
            'EMERG_VEH': int(data.get('EMERG_VEH', False)),
            'PASSENGER': int(data.get('PASSENGER', False)),
            'SPEEDING': int(data.get('SPEEDING', False)),
            'AG_DRIV': int(data.get('AG_DRIV', False)),
            'REDLIGHT': int(data.get('REDLIGHT', False)),
            'ALCOHOL': int(data.get('ALCOHOL', False)),
            'DISABILITY': int(data.get('DISABILITY', False)),
            'HOOD_158': str(data.get('HOOD_158', ''))
        }])
        
        # Transform the input data using the same pipeline
        transformed_data = transformer.transform(input_data)
        
        # Ensure columns match exactly with training data
        transformed_df = pd.DataFrame(transformed_data)
        print("Prediction Feature Names:", transformed_df.columns)


        # Make prediction
        prediction = knn_model.predict(transformed_df)
        prediction_proba = knn_model.predict_proba(transformed_df)
        
        response = {
            'prediction': str(prediction[0]),
            'confidence': float(max(prediction_proba[0])),
            'metrics': {
                'accuracy': 0.89,
                'precision': 0.91,
                'recall': 0.89,
                'f1_score': 0.90
}
        }
        
        return jsonify(response)
    
    except Exception as e:
        print("Prediction error:", str(e))
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')