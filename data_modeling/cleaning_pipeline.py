import pandas as pd

'''
This pipeline is responsible for cleaning the dataset before any transformation or model building,
by removing irrelevant columns, handling missing values, and extracting new features.
'''
class CleaningPipeline:
    # Columns with binary values identified in the dataset
    binary_columns = [
        'PEDESTRIAN',
        'CYCLIST',
        'AUTOMOBILE',
        'MOTORCYCLE',
        'TRUCK',
        'TRSN_CITY_VEH',
        'EMERG_VEH',
        'PASSENGER',
        'SPEEDING',
        'AG_DRIV',
        'REDLIGHT',
        'ALCOHOL',
        'DISABILITY'
    ]
    # List of relevant columns based on the context of the dataset, which should not be dropped
    relevant_columns = ['SPEEDING', 'REDLIGHT', 'ALCOHOL']
    # List of columns that are identifiers (ignoring ACCNUM at this point)
    identifier_columns = ['OBJECTID', 'INDEX']
    # List of columns that are not relevant for the model.
    irrelevant_columns = identifier_columns + [
        # The direction the vehicle was travelling to is not relevant
        'INITDIR',
        # We don't need to consider the exact streets where the collision happened, we already have other features for location
        'STREET1',
        'STREET2',
        # The police division is not relevant
        'DIVISION',
        # This is just a label for HOOD_158
        'NEIGHBOURHOOD_158',
        # The severity of the injury is just another way to describe the target variable
        'INJURY'
    ]
    # List of deprecated (old) columns specified in the documentation
    deprecated_columns = ['x', 'y', 'HOOD_140', 'NEIGHBOURHOOD_140']

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- CleaningPipeline ---")
        print("\nBefore cleaning")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())

        # Drop rows where ACCLASS is missing (target variable)
        df = df[df['ACCLASS'].notna()].copy()

        # Map ACCLASS to have only 'Fatal' and 'Non-Fatal'
        df['ACCLASS'] = df['ACCLASS'].replace({
            'Property Damage O': 'Non-Fatal',
            'Non-Fatal Injury': 'Non-Fatal',
            'Fatal': 'Fatal'
        })

        # Fill binary columns with 'No' where missing
        for col in self.binary_columns:
            df[col] = df[col].fillna('No')

        # Drop irrelevant columns and unique identifiers
        df = df.drop(columns=self.irrelevant_columns)

        print("\nIrrelevant columns dropped:", self.irrelevant_columns)

        # Drop deprecated columns
        df = df.drop(columns=self.deprecated_columns)

        print("\nDeprecated columns dropped:", self.deprecated_columns)

        # Drop columns where missing values exceed 80% (except identified relevant columns)
        before_dropping_max_nulls = df.shape[1]

        missing_values = df.isnull().mean()
        columns_to_drop = missing_values[missing_values > 0.8].index.tolist()
        columns_to_drop = [col for col in columns_to_drop if col not in self.relevant_columns]
        df = df.drop(columns=columns_to_drop)

        after_dropping_max_nulls = df.shape[1]

        print(f"\nDropped {before_dropping_max_nulls - after_dropping_max_nulls} columns with >80% missing values")
        print("Columns dropped:", columns_to_drop)

        # Handling columns with <3% missing values: If categorical, do not impute. Instead, discard rows with missing values.
        before_removing_nulls = df.shape[0]

        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].isnull().mean() < 0.03:
                df = df[df[col].notna()]
        
        after_removing_nulls = df.shape[0]

        print(f"\nRemoved {before_removing_nulls - after_removing_nulls} rows with <3% missing values on categorical columns")

        # Extract day, month, weekday from 'DATE'
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df['DAY'] = df['DATE'].dt.day
        df['MONTH'] = df['DATE'].dt.month
        df['WEEKDAY'] = df['DATE'].dt.day_name()
        df = df.drop(columns=['DATE'])

        # Since the data includes every person involved in a collision event, ACCNUM is duplicated.
        # We don't want to keep only one row per ACCNUM, as we would lose valuable information.
        # Only drop rows that are completely duplicated:
        before_removing_duplicates = df.shape[0]
        df = df.drop_duplicates()
        after_removing_duplicates = df.shape[0]
        print(f"\nRemoved {before_removing_duplicates - after_removing_duplicates} completely duplicated rows")

        # Remove ACCNUM column
        df = df.drop(columns=['ACCNUM'])

        print("\nDropped ACCNUM column")

        print("\nAfter Cleaning")
        print("Shape:", df.shape)
        print("Remaining Columns:", df.columns.tolist())

        return df
