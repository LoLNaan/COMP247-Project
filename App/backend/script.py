from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Helper function to generate random metrics
def generate_random_metrics(model_name):
    base_metrics = {
        "accuracy": round(random.random(), 4),
        "precision": round(random.random(), 4),
        "recall": round(random.random(), 4),
        "f1_score": round(random.random(), 4),
        "confusion_matrix": {
            "true_positive": random.randint(0, 100),
            "true_negative": random.randint(0, 100),
            "false_positive": random.randint(0, 50),
            "false_negative": random.randint(0, 50),
        },
        "prediction": "Fatal" if random.random() > 0.5 else "Non-Fatal",
        "prediction_probability": round(random.random(), 4),
        "model_used": model_name,
        "timestamp": datetime.now().isoformat(),
    }

    # Add model-specific metrics
    if 'logistic' in model_name.lower():
        base_metrics["coefficients"] = [round(random.uniform(-1, 1), 4) for _ in range(5)]
    elif 'decision' in model_name.lower():
        base_metrics["tree_depth"] = random.randint(3, 12)
        base_metrics["feature_importance"] = {
            "TIME": round(random.random(), 4),
            "ROAD_CLASS": round(random.random(), 4),
            "DISTRICT": round(random.random(), 4),
            "VISIBILITY": round(random.random(), 4),
            "LIGHT": round(random.random(), 4),
        }
    elif 'svm' in model_name.lower():
        base_metrics["support_vectors"] = random.randint(50, 150)
        base_metrics["kernel"] = "rbf"
    elif 'random' in model_name.lower():
        base_metrics["num_trees"] = random.randint(50, 150)
        base_metrics["oob_score"] = round(random.random(), 4)
    elif 'neural' in model_name.lower() or 'predict' in model_name.lower():
        base_metrics["layers"] = 3
        base_metrics["activation"] = "relu"
        base_metrics["learning_rate"] = round(random.random() * 0.1, 4)

    return base_metrics

@app.route('/api/models/logistic-regression/predict', methods=['POST'])
def logistic_regression():
    print("Received data for Logistic Regression:", request.json)
    time.sleep(1.5)  # Simulate processing delay
    return jsonify(generate_random_metrics("Logistic Regression"))

@app.route('/api/models/decision-tree/predict', methods=['POST'])
def decision_tree():
    print("Received data for Decision Tree:", request.json)
    time.sleep(1.0)
    return jsonify(generate_random_metrics("Decision Tree"))

@app.route('/api/models/svm/predict', methods=['POST'])
def svm():
    print("Received data for SVM:", request.json)
    time.sleep(2.0)
    return jsonify(generate_random_metrics("Support Vector Machine"))

@app.route('/api/models/random-forest/predict', methods=['POST'])
def random_forest():
    print("Received data for Random Forest:", request.json)
    time.sleep(1.2)
    return jsonify(generate_random_metrics("Random Forest"))

@app.route('/api/models/knn/predict', methods=['POST'])
def knn():
    print("Received data for KNN:", request.json)
    time.sleep(0.8)
    return jsonify(generate_random_metrics("K-Nearest Neighbors"))

@app.route('/api/models/predict', methods=['POST'])
def default_model():
    print("Received data for generic model:", request.json)
    time.sleep(1.0)
    return jsonify(generate_random_metrics("Default Model"))

if __name__ == '__main__':
    app.run(port=5000, debug=True)