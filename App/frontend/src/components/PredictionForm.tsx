import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Brain, LineChart, Trees, ScatterChart, Divide} from "lucide-react";

type ModelType = "logreg" | "neural_network" | "svm" | "rf" | "knn";

interface FormData {

  // Numeric/Text Fields
  TIME: string;
  ROAD_CLASS: string;
  DISTRICT: string;
  LATITUDE: string;
  LONGITUDE: string;
  ACCLOC: string;
  TRAFFCTL: string;
  VISIBILITY: string;
  LIGHT: string;
  RDSFCOND: string;
  IMPACTYPE: string;
  INVTYPE: string;
  INVAGE: string;
  INJURY: string;
  INITDIR: string;
  VEHTYPE: string;
  MANOEUVER: string;
  DRIVACT: string;
  DRIVCOND: string;
  DAY: string;
  MONTH: string;
  WEEKDAY: string;
  // Boolean Fields
  PEDESTRIAN: boolean;
  CYCLIST: boolean;
  AUTOMOBILE: boolean;
  MOTORCYCLE: boolean;
  TRUCK: boolean;
  TRSN_CITY_VEH: boolean;
  EMERG_VEH: boolean;
  PASSENGER: boolean;
  SPEEDING: boolean;
  AG_DRIV: boolean;
  REDLIGHT: boolean;
  ALCOHOL: boolean;
  DISABILITY: boolean;
  HOOD_158: string;
}

function PredictionForm() {
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState<ModelType>("logreg");
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState<FormData>({
    TIME: "",
    ROAD_CLASS: "",
    DISTRICT: "",
    LATITUDE: "",
    LONGITUDE: "",
    ACCLOC: "",
    TRAFFCTL: "",
    VISIBILITY: "",
    LIGHT: "",
    RDSFCOND: "",
    IMPACTYPE: "",
    INVTYPE: "",
    INVAGE: "",
    INJURY: "",
    INITDIR: "",
    VEHTYPE: "",
    MANOEUVER: "",
    DRIVACT: "",
    DRIVCOND: "",
    DAY: "",
    MONTH: "",
    WEEKDAY: "",
    PEDESTRIAN: false,
    CYCLIST: false,
    AUTOMOBILE: false,
    MOTORCYCLE: false,
    TRUCK: false,
    TRSN_CITY_VEH: false,
    EMERG_VEH: false,
    PASSENGER: false,
    SPEEDING: false,
    AG_DRIV: false,
    REDLIGHT: false,
    ALCOHOL: false,
    DISABILITY: false,
    HOOD_158: "",
  });

  const models = [
    {
      id: "logreg",
      name: "Logistic Regression",
      icon: LineChart,
      endpoint: "http://localhost:5000/predict",
    },
    {
      id: "neural_network",
      name: "Neural Networks",
      icon: Brain,
      endpoint: "http://localhost:5000/predict",
    },
    {
      id: "svm",
      name: "Support Vector Machine",
      icon: Divide,
      endpoint: "http://localhost:5000/predict",
    },
    {
      id: "rf",
      name: "Random Forest",
      icon: Trees,
      endpoint: "http://localhost:5000/predict",
    },
    {
      id: "knn",
      name: "K nearest neighbourest",
      icon: ScatterChart,
      endpoint: "http://localhost:5000/predict",
    },
  ];

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const target = e.target;
    const value =
      target instanceof HTMLInputElement && target.type === "checkbox"
        ? target.checked
        : target.value;
    const name = target.name;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      // Get the correct endpoint for the selected model
      const selectedModelData = models.find(
        (model) => model.id === selectedModel
      );
      const apiEndpoint = selectedModelData?.endpoint || "/api/models/predict";

      // Make API call to the selected model's endpoint
      const response = await fetch(apiEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // The backend doesn't actually use the form data, but we'll send it anyway
        // to maintain the POST request structure
        // body: JSON.stringify({
        //   model: "rf",
        //   LATITUDE: 43.763491,
        //   LONGITUDE: -79.301257,
        //   TIME: 2125,
        //   ROAD_CLASS: "Major Arterial",
        //   DISTRICT: "Scarborough",
        //   ACCLOC: "At Intersection",
        //   TRAFFCTL: "No Control",
        //   VISIBILITY: "Clear",
        //   LIGHT: "Dark",
        //   RDSFCOND: "Dry",
        //   IMPACTYPE: "Pedestrian Collisions",
        //   INVTYPE: "Pedestrian",
        //   INVAGE: "30 to 34",
        //   INITDIR: "East",
        //   VEHTYPE: "Other",
        //   MANOEUVER: "Other",
        //   DRIVACT: "Other",
        //   DRIVCOND: "Other",
        //   PEDESTRIAN: true,
        //   CYCLIST: false,
        //   AUTOMOBILE: true,
        //   MOTORCYCLE: false,
        //   TRUCK: false,
        //   TRSN_CITY_VEH: false,
        //   EMERG_VEH: false,
        //   PASSENGER: false,
        //   SPEEDING: false,
        //   AG_DRIV: true,
        //   REDLIGHT: true,
        //   WEEKDAY: "Wednesday",
        // }),
        body: JSON.stringify({
          model: selectedModel,
          LATITUDE: formData.LATITUDE,
          LONGITUDE: formData.LONGITUDE,
          TIME: formData.TIME,
          ROAD_CLASS: formData.ROAD_CLASS,
          DISTRICT: formData.DISTRICT,
          ACCLOC: formData.ACCLOC,
          TRAFFCTL: formData.TRAFFCTL,
          VISIBILITY: formData.VISIBILITY,
          LIGHT: formData.LIGHT,
          RDSFCOND: formData.RDSFCOND,
          IMPACTYPE: formData.IMPACTYPE,
          INVTYPE: formData.INVTYPE,
          INVAGE: formData.INVAGE,
          INITDIR: formData.INITDIR,
          VEHTYPE: formData.VEHTYPE,
          MANOEUVER: formData.MANOEUVER,
          DRIVACT: formData.DRIVACT,
          DRIVCOND: formData.DRIVCOND,
          PEDESTRIAN: formData.PEDESTRIAN,
          CYCLIST: formData.CYCLIST,
          AUTOMOBILE: formData.AUTOMOBILE,
          MOTORCYCLE: formData.MOTORCYCLE,
          TRUCK: formData.TRUCK,
          TRSN_CITY_VEH: formData.TRSN_CITY_VEH,
          EMERG_VEH: formData.EMERG_VEH,
          PASSENGER: formData.PASSENGER,
          SPEEDING: formData.SPEEDING,
          AG_DRIV: formData.AG_DRIV,
          REDLIGHT: formData.REDLIGHT,
          WEEKDAY: formData.WEEKDAY,
        }),
      });

      if (!response.ok) {
        throw new Error(`API call failed with status: ${response.status}`);
      }

      const data = await response.json();

      // Navigate to results page with the response data
      navigate("/results", { state: { metrics: data } });
    } catch (error) {
      console.error("Error making prediction:", error);

      alert("Failed to make prediction. Please check the console for details.");
    } finally {
      setIsLoading(false);
    }
  };

  // Handler to navigate to home page
  const goToHome = () => {
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      {/* Loading Overlay with Brain Animation */}
      {isLoading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-8 rounded-lg shadow-xl flex flex-col items-center">
            <div className="relative mb-4">
              <Brain className="w-16 h-16 text-indigo-600" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div
                  className="absolute w-20 h-20 rounded-full bg-indigo-200 opacity-50 animate-ping"
                  style={{ animationDuration: "1.5s" }}
                ></div>
                <div
                  className="absolute w-16 h-16 rounded-full bg-indigo-300 opacity-30 animate-ping"
                  style={{ animationDuration: "2s" }}
                ></div>
                <div
                  className="absolute w-12 h-12 rounded-full bg-indigo-400 opacity-20 animate-ping"
                  style={{ animationDuration: "2.5s" }}
                ></div>
              </div>
              <div className="absolute inset-0">
                <div className="absolute top-1 left-4 w-2 h-2 bg-indigo-600 rounded-full animate-pulse"></div>
                <div
                  className="absolute top-3 right-4 w-2 h-2 bg-indigo-600 rounded-full animate-pulse"
                  style={{ animationDelay: "0.3s" }}
                ></div>
                <div
                  className="absolute bottom-3 left-5 w-2 h-2 bg-indigo-600 rounded-full animate-pulse"
                  style={{ animationDelay: "0.6s" }}
                ></div>
                <div
                  className="absolute bottom-1 right-5 w-2 h-2 bg-indigo-600 rounded-full animate-pulse"
                  style={{ animationDelay: "0.9s" }}
                ></div>
                <div
                  className="absolute top-8 left-2 w-2 h-2 bg-indigo-600 rounded-full animate-pulse"
                  style={{ animationDelay: "1.2s" }}
                ></div>
              </div>
            </div>
            <p className="text-lg font-medium text-gray-700">
              {models.find((m) => m.id === selectedModel)?.name} Thinking...
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Processing your data with{" "}
              {models.find((m) => m.id === selectedModel)?.name}
            </p>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="bg-indigo-600 text-white py-4 px-6">
        <div className="max-w-7xl mx-auto flex items-center">
          <Brain
            className="w-6 h-6 mr-2 cursor-pointer hover:text-indigo-200 transition-colors"
            onClick={goToHome}
          />
          <h1 className="text-xl font-semibold">
            Machine Learning - KSI Dataset
          </h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <form onSubmit={handleSubmit}>
          {/* Model Selection */}
          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              Choose the model
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
              {models.map((model) => {
                const Icon = model.icon;
                return (
                  <button
                    type="button"
                    key={model.id}
                    onClick={() => setSelectedModel(model.id as ModelType)}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      selectedModel === model.id
                        ? "border-indigo-600 bg-indigo-50"
                        : "border-gray-200 hover:border-indigo-200"
                    }`}
                  >
                    <Icon
                      className={`w-8 h-8 mb-2 mx-auto ${
                        selectedModel === model.id
                          ? "text-indigo-600"
                          : "text-gray-500"
                      }`}
                    />
                    <p
                      className={`text-center font-medium ${
                        selectedModel === model.id
                          ? "text-indigo-600"
                          : "text-gray-700"
                      }`}
                    >
                      {model.name}
                    </p>
                  </button>
                );
              })}
            </div>
          </section>

          {/* Input Form */}
          <section className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-6">
              Model Parameters
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Numeric / Text Inputs */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  TIME
                </label>
                <input
                  type="number"
                  name="TIME"
                  value={formData.TIME}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Like 2125"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  ROAD_CLASS
                </label>
                <input
                  type="text"
                  name="ROAD_CLASS"
                  value={formData.ROAD_CLASS}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter road class"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  DISTRICT
                </label>
                <input
                  type="text"
                  name="DISTRICT"
                  value={formData.DISTRICT}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter district"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  LATITUDE
                </label>
                <input
                  type="number"
                  step="any"
                  name="LATITUDE"
                  value={formData.LATITUDE}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter latitude"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  LONGITUDE
                </label>
                <input
                  type="number"
                  step="any"
                  name="LONGITUDE"
                  value={formData.LONGITUDE}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter longitude"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  ACCLOC
                </label>
                <input
                  type="text"
                  name="ACCLOC"
                  value={formData.ACCLOC}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter accident location"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  TRAFFCTL
                </label>
                <input
                  type="text"
                  name="TRAFFCTL"
                  value={formData.TRAFFCTL}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter traffic control"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  VISIBILITY
                </label>
                <input
                  type="text"
                  name="VISIBILITY"
                  value={formData.VISIBILITY}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Clear, Rain, etc."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  LIGHT
                </label>
                <input
                  type="text"
                  name="LIGHT"
                  value={formData.LIGHT}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Daylight, Dark, etc."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  RDSFCOND
                </label>
                <input
                  type="text"
                  name="RDSFCOND"
                  value={formData.RDSFCOND}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Dry, Wet, etc."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  IMPACTYPE
                </label>
                <input
                  type="text"
                  name="IMPACTYPE"
                  value={formData.IMPACTYPE}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Rear End, Sideswipe, etc."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  INVTYPE
                </label>
                <input
                  type="text"
                  name="INVTYPE"
                  value={formData.INVTYPE}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Driver, Pedestrian, etc."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  INVAGE
                </label>
                <input
                  type="text"
                  name="INVAGE"
                  value={formData.INVAGE}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter age"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  INJURY
                </label>
                <input
                  type="text"
                  name="INJURY"
                  value={formData.INJURY}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Major, Minor, etc.."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  INITDIR
                </label>
                <input
                  type="text"
                  name="INITDIR"
                  value={formData.INITDIR}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="North, South, East etc.."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  VEHTYPE
                </label>
                <input
                  type="text"
                  name="VEHTYPE"
                  value={formData.VEHTYPE}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Automobile, Passenget, Other etc.."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  MANOEUVER
                </label>
                <input
                  type="text"
                  name="MANOEUVER"
                  value={formData.MANOEUVER}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter maneuver type Eg: Going Ahead"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  DRIVACT
                </label>
                <input
                  type="text"
                  name="DRIVACT"
                  value={formData.DRIVACT}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter driver action"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  DRIVCOND
                </label>
                <input
                  type="text"
                  name="DRIVCOND"
                  value={formData.DRIVCOND}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter driver condition"
                  required
                />
              </div>

              {/* Boolean / Checkbox Inputs */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  PEDESTRIAN
                </label>
                <input
                  type="checkbox"
                  name="PEDESTRIAN"
                  checked={formData.PEDESTRIAN}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  CYCLIST
                </label>
                <input
                  type="checkbox"
                  name="CYCLIST"
                  checked={formData.CYCLIST}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  AUTOMOBILE
                </label>
                <input
                  type="checkbox"
                  name="AUTOMOBILE"
                  checked={formData.AUTOMOBILE}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  MOTORCYCLE
                </label>
                <input
                  type="checkbox"
                  name="MOTORCYCLE"
                  checked={formData.MOTORCYCLE}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  TRUCK
                </label>
                <input
                  type="checkbox"
                  name="TRUCK"
                  checked={formData.TRUCK}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  TRSN_CITY_VEH
                </label>
                <input
                  type="checkbox"
                  name="TRSN_CITY_VEH"
                  checked={formData.TRSN_CITY_VEH}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  EMERG_VEH
                </label>
                <input
                  type="checkbox"
                  name="EMERG_VEH"
                  checked={formData.EMERG_VEH}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  PASSENGER
                </label>
                <input
                  type="checkbox"
                  name="PASSENGER"
                  checked={formData.PASSENGER}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  SPEEDING
                </label>
                <input
                  type="checkbox"
                  name="SPEEDING"
                  checked={formData.SPEEDING}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  AG_DRIV
                </label>
                <input
                  type="checkbox"
                  name="AG_DRIV"
                  checked={formData.AG_DRIV}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  REDLIGHT
                </label>
                <input
                  type="checkbox"
                  name="REDLIGHT"
                  checked={formData.REDLIGHT}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  ALCOHOL
                </label>
                <input
                  type="checkbox"
                  name="ALCOHOL"
                  checked={formData.ALCOHOL}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  DISABILITY
                </label>
                <input
                  type="checkbox"
                  name="DISABILITY"
                  checked={formData.DISABILITY}
                  onChange={handleInputChange}
                  className="mr-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  HOOD_158
                </label>
                <input
                  type="text"
                  name="HOOD_158"
                  value={formData.HOOD_158}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="HOOD_158"
                  required
                />
              </div>
              {/* Additional Numeric Inputs for day, month, weekday */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  DAY
                </label>
                <input
                  type="number"
                  name="DAY"
                  value={formData.DAY}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="1-31"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  MONTH
                </label>
                <input
                  type="number"
                  name="MONTH"
                  value={formData.MONTH}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="1-12"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  WEEKDAY
                </label>
                <input
                  type="text"
                  name="WEEKDAY"
                  value={formData.WEEKDAY}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Sunday - Saturday"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="mt-8 bg-indigo-600 text-white px-6 py-2 rounded-md hover:bg-indigo-700 transition-colors disabled:bg-indigo-400"
            >
              {isLoading ? "Processing..." : "Make Prediction"}
            </button>
          </section>
        </form>
      </main>
    </div>
  );
}

export default PredictionForm;
