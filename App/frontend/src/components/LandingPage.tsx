import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, ArrowRight, Award, Cpu, LineChart, Binary } from 'lucide-react';

function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-b from-indigo-50 via-white to-indigo-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="relative pt-16 pb-20">
            <div className="text-center">
              <Brain className="w-16 h-16 mx-auto text-indigo-600 mb-6" />
              <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
                KSI Dataset
                <span className="text-indigo-600"> Prediction</span>
              </h1>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
                Leverage the power of machine learning to predict accident outcomes using multiple advanced algorithms.
              </p>
              <button
                onClick={() => navigate('/predict')}
                className="inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 transition-colors"
              >
                Start Predicting
                <ArrowRight className="ml-2 w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900">Advanced ML Models</h2>
            <p className="mt-4 text-lg text-gray-600">Choose from multiple algorithms for optimal predictions</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow">
              <Binary className="w-10 h-10 text-indigo-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">5 Different Models</h3>
              <p className="text-gray-600">Try out 5 different models to predict severity of accidents.</p>
            </div>

            <div className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow">
              <LineChart className="w-10 h-10 text-indigo-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Statistical Models</h3>
              <p className="text-gray-600">Traditional algorithms with proven track records</p>
            </div>

            <div className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow">
              <Cpu className="w-10 h-10 text-indigo-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Real-time Processing</h3>
              <p className="text-gray-600">Fast and efficient prediction generation</p>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="py-16 bg-indigo-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold text-indigo-600 mb-2">High Accuracy</div>
              <div className="text-gray-600">All models were finetuned to make prefect predictions.</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-indigo-600 mb-2">Real Time Data</div>
              <div className="text-gray-600">From toronto police</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-indigo-600 mb-2">5</div>
              <div className="text-gray-600">ML Models</div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <Award className="w-12 h-12 text-indigo-600 mx-auto mb-6" />
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Ready to make predictions?
          </h2>
          <p className="text-xl text-gray-600 mb-8">
            Start using our advanced machine learning models today
          </p>
          <button
            onClick={() => navigate('/predict')}
            className="inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 transition-colors"
          >
            Get Started
            <ArrowRight className="ml-2 w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;