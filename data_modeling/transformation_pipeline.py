import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

'''
Custom mapping transformer to simplify categorical features and group low-frequency values.
'''
class CategoryMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # The following code was used to identify unique values in each categorical column and map similar values together:
        # # Get only object (categorical) columns
        # cat_cols = df.select_dtypes(include="object").columns

        # # Print unique values for each categorical column
        # for col in cat_cols:
        #     print(f"🔹 {col} - Unique values:")
        #     print(df[col].dropna().unique())
        #     print("-" * 40)

        # Simplify DRIVCOND to 'Alcohol', 'Fatigued', 'Medical', 'Other', 'Unknown'
        df['DRIVCOND'] = df['DRIVCOND'].replace({
            'Ability Impaired, Alcohol': 'Alcohol',
            'Ability Impaired, Alcohol Over .08': 'Alcohol',
            'Ability Impaired, Drugs': 'Alcohol',
            'Had Been Drinking': 'Alcohol',
            'Fatigue': 'Fatigued',
            'Medical or Physical Disability': 'Medical',
            'Unknown': 'Unknown',
            'Other': 'Other'
        })

        # Simplify VEHTYPE to common types to 'Automobile', 'Truck', 'Bus', 'Motorcycle', 'Transit', 'Emergency', 'Bicycle', 'Other', 'Unknown'
        df['VEHTYPE'] = df['VEHTYPE'].replace({
            'Automobile, Station Wagon': 'Automobile',
            'Passenger Van': 'Automobile',
            'Pick Up Truck': 'Truck',
            'Truck - Open': 'Truck',
            'Truck - Closed (Blazer, etc)': 'Truck',
            'Truck - Dump': 'Truck',
            'Truck (other)': 'Truck',
            'Truck-Tractor': 'Truck',
            'Truck - Tank': 'Truck',
            'Truck - Car Carrier': 'Truck',
            'Delivery Van': 'Truck',
            'Municipal Transit Bus (TTC)': 'Bus',
            'Bus (Other) (Go Bus, Gray Coa': 'Bus',
            'Intercity Bus': 'Bus',
            'School Bus': 'Bus',
            'Motorcycle': 'Motorcycle',
            'Moped': 'Motorcycle',
            'Off Road - 2 Wheels': 'Motorcycle',
            'Off Road - 4 Wheels': 'Motorcycle',
            'Street Car': 'Transit',
            'Police Vehicle': 'Emergency',
            'Fire Vehicle': 'Emergency',
            'Ambulance': 'Emergency',
            'Other Emergency Vehicle': 'Emergency',
            'Bicycle': 'Bicycle',
            'Rickshaw': 'Other',
            'Construction Equipment': 'Other',
            'Other': 'Other',
            'Unknown': 'Unknown'
        })

        # Simplify DRIVACT to 'Speeding', 'Slow', 'Proper', 'Other'
        df['DRIVACT'] = df['DRIVACT'].replace({
            'Speed too Fast For Condition': 'Speeding',
            'Exceeding Speed Limit': 'Speeding',
            'Speed too Slow': 'Slow',
            'Driving Properly': 'Proper',
            'Other': 'Other',
            'Unknown': 'Other'
        })

        # Simplify INVAGE to 'Child', 'Teen', 'Young Adult', 'Adult', 'Senior', 'Elderly', 'Unknown' (broader age groups)
        age_mapping = {
            '0 to 4': 'Child',
            '5 to 9': 'Child',
            '10 to 14': 'Teen',
            '15 to 19': 'Teen',
            '20 to 24': 'Young Adult',
            '25 to 29': 'Young Adult',
            '30 to 34': 'Adult',
            '35 to 39': 'Adult',
            '40 to 44': 'Adult',
            '45 to 49': 'Adult',
            '50 to 54': 'Senior',
            '55 to 59': 'Senior',
            '60 to 64': 'Senior',
            '65 to 69': 'Elderly',
            '70 to 74': 'Elderly',
            '75 to 79': 'Elderly',
            '80 to 84': 'Elderly',
            '85 to 89': 'Elderly',
            '90 to 94': 'Elderly',
            'Over 95': 'Elderly',
            'unknown': 'Unknown'
        }
        df['INVAGE'] = df['INVAGE'].map(age_mapping)

        return df

# Define the full transformation pipeline
class TransformationPipeline:
    def __init__(self):
        self.pipeline = None
        self.feature_names = None
        self.preprocessor = None

    def build_pipeline(self, X: pd.DataFrame):
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Remove target column in case the full dataset was passed
        if 'ACCLASS' in categorical_cols:
            categorical_cols.remove('ACCLASS')

        # Preprocessor: scale numerical, impute nan values, encode categorical
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_cols)
            ]
        )

        # Full pipeline: first mapping, then preprocessing
        self.pipeline = Pipeline([
            ('mapper', CategoryMapper()),
            ('preprocessor', self.preprocessor)
        ])

        return self.pipeline

    def fit(self, X, y=None):
        self.build_pipeline(X)
        self.pipeline.fit(X, y)
        # Store feature names after fit
        cat_cols = self.preprocessor.transformers_[1][2]
        encoder = self.preprocessor.named_transformers_['cat']
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        self.feature_names = np.concatenate([self.preprocessor.transformers_[0][2], encoded_cols])
        return self

    def transform(self, X):
        X_transformed = self.pipeline.transform(X)
        return pd.DataFrame(X_transformed, columns=self.feature_names)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        return self.feature_names