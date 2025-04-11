import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

interface ResponseData {
  prediction: string;
  confidence: number;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
}

function ArcProgress({ value, label, color }: { value: number; label: string; color: string }) {
  const radius = 60;
  const strokeWidth = 10;
  const normalizedValue = Math.min(Math.max(value, 0), 1);
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - normalizedValue);

  return (
    <div className="flex flex-col items-center">
      <div className="relative">
        <svg width="150" height="150" className="-rotate-90">
          <circle
            cx="75"
            cy="75"
            r={radius}
            stroke="#e5e7eb"
            strokeWidth={strokeWidth}
            fill="none"
          />
          <circle
            cx="75"
            cy="75"
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold">{Math.round(value * 100)}%</span>
        </div>
      </div>
      <span className="mt-4 text-lg font-medium text-gray-700">{label}</span>
    </div>
  );
}

function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  const responseData = location.state?.metrics as ResponseData;

  if (!responseData) {
    navigate("/");
    return null;
  }

  const { prediction, confidence, metrics } = responseData;

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      <header className="bg-indigo-600 text-white py-4 px-6">
        <div className="max-w-7xl mx-auto flex items-center">
          <button
            onClick={() => navigate("/")}
            className="flex items-center text-white hover:text-indigo-200 transition-colors"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back to Prediction
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12 space-y-8">
        {/* Prediction Results Card */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
            Prediction Results
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-indigo-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold text-indigo-800 mb-2">
                Prediction
              </h3>
              <p className="text-2xl font-bold text-gray-800">{prediction}</p>
            </div>
            <div className="bg-indigo-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold text-indigo-800 mb-2">
                Confidence
              </h3>
              <p className="text-2xl font-bold text-gray-800">
                {Math.round(confidence * 100)}%
              </p>
            </div>
          </div>
        </div>

        {/* Metrics Card */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">
            Model Performance Metrics
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
            <ArcProgress
              value={metrics.accuracy}
              label="Accuracy"
              color="#4f46e5"
            />
            <ArcProgress
              value={metrics.precision}
              label="Precision"
              color="#06b6d4"
            />
            <ArcProgress
              value={metrics.recall}
              label="Recall"
              color="#0891b2"
            />
            <ArcProgress
              value={metrics.f1_score}
              label="F1 Score"
              color="#10b981"
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default Results;