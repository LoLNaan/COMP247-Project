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
        #     print(f"{col} - Unique values:")
        #     print(df[col].dropna().unique())
        #     print("-" * 40)

        # Simplify DRIVCOND grouping similar values together
        driver_condition_mapping = {
            'Normal': 'Normal',
            'Inattentive': 'Inattentive',
            'Ability Impaired, Alcohol': 'Alcohol',
            'Ability Impaired, Alcohol Over .08': 'Alcohol',
            'Had Been Drinking': 'Alcohol',
            'Ability Impaired, Drugs': 'Drugs',
            'Fatigue': 'Fatigue',
            'Medical or Physical Disability': 'Medical',
            'Other': 'Other',
            'Unknown': 'Other'
        }
        df['DRIVCOND'] = df['DRIVCOND'].replace(driver_condition_mapping)

        # Simplify ACCLOC to 'Private', 'Intersection', 'Other'
        accident_location_mapping = {
            'At/Near Private Drive': 'Private',
            'Private Driveway': 'Private',
            'Overpass or Bridge': 'Overpass or Bridge',
            'At Intersection': 'Intersection',
            'Intersection Related': 'Intersection',
            'Non Intersection': 'Other',
            'Laneway': 'Other',
            'Other': 'Other',
        }
        df['ACCLOC'] = df['ACCLOC'].replace(accident_location_mapping)

        # Simplify VISIBILITY redundant categories
        visibility_mapping = {
            'Clear': 'Clear',
            'Rain': 'Rain',
            'Freezing Rain': 'Rain',
            'Snow': 'Snow',
            'Drifting Snow': 'Snow',
            'Fog, Mist, Smoke, Dust': 'Fog',
            'Strong wind': 'Wind',
            'Other': 'Other',
        }
        df['VISIBILITY'] = df['VISIBILITY'].replace(visibility_mapping)

        # Simplify LIGHT redundant categories
        light_mapping = {
            'Daylight': 'Daylight',
            'Daylight, artificial': 'Daylight',
            'Dark': 'Dark',
            'Dark, artificial': 'Dark',
            'Dawn': 'Dawn',
            'Dawn, artificial': 'Dawn',
            'Dusk': 'Dusk',
            'Dusk, artificial': 'Dusk',
        }
        df['LIGHT'] = df['LIGHT'].replace(light_mapping)

        # Simplify RDSFCOND to 'Dry', 'Wet', 'Snow', 'Ice', 'Other'
        road_condition_mapping = {
            'Dry': 'Dry',
            'Wet': 'Wet',
            'Slush': 'Wet',
            'Spilled liquid': 'Wet',
            'Loose Snow': 'Snow',
            'Packed Snow': 'Snow',
            'Ice': 'Ice',
            'Loose Sand or Gravel': 'Other',
            'Other': 'Other',
        }
        df['RDSFCOND'] = df['RDSFCOND'].replace(road_condition_mapping)

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

'''
Custom full transformation pipeline.
It uses the CategoryMapper defined above first, and then preprocesses numerical and categorical columns separately.
'''
class TransformationPipeline:
    def __init__(self):
        self.pipeline = None
        self.feature_names = None
        self.preprocessor = None

    def build_pipeline(self, X: pd.DataFrame):
        print("\n--- TransformationPipeline ---")
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                # Transformer pipeline for numerical columns
                ('num', Pipeline([
                    # Impute missing values with the mean
                    ('imputer', SimpleImputer(strategy='mean')),
                    # Scale the data
                    ('scaler', StandardScaler())
                ]), numerical_cols),
                # Transformer pipeline for categorical columns
                ('cat', Pipeline([
                    # Impute missing values with the most frequent value
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    # One-hot encode the data
                    # handle_unknown='ignore' to ignore unseen values during prediction, sparse_output=False to return a dense matrix
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_cols)
            ]
        )

        # Full pipeline
        self.pipeline = Pipeline([
            # Apply the custom category mapper
            ('mapper', CategoryMapper()),
            # Apply the preprocessor
            ('preprocessor', self.preprocessor)
        ])

        return self.pipeline

    def fit(self, X, y=None):
        print("\nFitting the transformation pipeline...")
        # Build and fit the pipeline
        self.build_pipeline(X)
        self.pipeline.fit(X, y)
        # Get the numerical and categorical columns from the preprocessor
        num_cols = self.preprocessor.transformers_[0][2]
        cat_cols = self.preprocessor.transformers_[1][2]
        # Get the encoder from the preprocessor
        encoder = self.preprocessor.named_transformers_['cat']
        # Get the one-hot encoded column names
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        # Combine the numerical columns with the encoded categorical columns
        self.feature_names = np.concatenate([num_cols, encoded_cols])

        return self

    def transform(self, X):
        print("\nTransforming the data...")
        # Transform the data
        X_transformed = self.pipeline.transform(X)

        # Print the number of features after transformation
        print(f"\nNumber of features after transformation: {X_transformed.shape[1]}")

        # Return a DataFrame with the transformed data and feature names
        return pd.DataFrame(X_transformed, columns=self.feature_names)

    def fit_transform(self, X, y=None):
        # Fit and transform the data
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        # Return the feature names
        return self.feature_names