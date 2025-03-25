import numpy as np
import pandas as pd

'''
This pipeline is responsible for cleaning the dataset before any transformation or model building,
by removing irrelevant columns, handling missing values, and extracting new features.
'''
class CleaningPipeline:
    # List of columns that are identifiers (ignoring ACCNUM at this point)
    identifier_columns = ['OBJECTID', 'INDEX']
    # List of columns manually selected based on the context of the dataset, which should not be dropped
    relevant_columns = ['SPEEDING', 'REDLIGHT', 'ALCOHOL']
    # List of columns manually selected that are not relevant for the model.
    irrelevant_columns = [
        # We don't need to consider the exact streets where the collision happened, we already have both the exact coordinates and the neighbourhood
        'STREET1',
        'STREET2',
        # The police division is not relevant or related to the target variable
        'DIVISION',
        # This is just a label for HOOD_158
        'NEIGHBOURHOOD_158',
    ]
    # List of deprecated (old) columns specified in the documentation
    deprecated_columns = ['HOOD_140', 'NEIGHBOURHOOD_140']

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- CleaningPipeline ---")
        print("\nBefore cleaning")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())

        # Drop rows where ACCLASS is missing (target variable)
        before_removing_acclass_nulls = df.shape[0]
        df = df[df['ACCLASS'].notna()].copy()
        after_removing_acclass_nulls = df.shape[0]
        print(f"\nRemoved {before_removing_acclass_nulls - after_removing_acclass_nulls} rows with missing ACCLASS")

        # Map ACCLASS to have only 'Fatal' and 'Non-Fatal'
        df['ACCLASS'] = df['ACCLASS'].replace({
            'Property Damage O': 'Non-Fatal',
            'Non-Fatal Injury': 'Non-Fatal',
            'Fatal': 'Fatal'
        })

        # Drop identifier columns
        df = df.drop(columns=self.identifier_columns)
        print(f"\nDropped {len(self.identifier_columns)} identifier columns: {self.identifier_columns}")

        # Identify binary columns: values 'Yes', 'No', or NaN only
        binary_columns = []
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({'Yes', 'No'}):
                binary_columns.append(col)

        print(f"\nFill missing 'No' in {len(binary_columns)} detected binary columns: {binary_columns}")

        # Fill binary columns with 'No' where missing
        for col in binary_columns:
            df[col] = df[col].fillna('No')

        # Drop completely duplicated rows
        before_removing_duplicates = df.shape[0]
        df = df.drop_duplicates()
        after_removing_duplicates = df.shape[0]
        print(f"\nRemoved {before_removing_duplicates - after_removing_duplicates} completely duplicated rows (just ignoring unique identifiers)")

        # Remove ACCNUM column
        df = df.drop(columns=['ACCNUM'])

        print("\nDropped ACCNUM column")

        # Detect columns with identical values
        duplicated_columns = []
        cols = df.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if df[cols[i]].equals(df[cols[j]]):
                    print(f"Duplicate columns found: {cols[i]} == {cols[j]}")
                    duplicated_columns.append(cols[j])

        df.drop(columns=duplicated_columns, inplace=True)

        if len(duplicated_columns) == 0:
            print("\nNo columns with identical values found.")
        else:
            print(f"\nDropped {len(duplicated_columns)} columns with identical values: {duplicated_columns}")

        # Calculate the absolute correlation matrix for numerical columns
        corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

        # Keep only the upper triangle of the matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find pairs of columns with high correlation
        correlation_threshold = 0.95
        high_corr_pairs = [
            (col, upper[col].idxmax(), upper[col].max())
            for col in upper.columns
            if upper[col].max() > correlation_threshold
        ]

        if len(high_corr_pairs) == 0:
            print("\nNo highly correlated columns detected.")
        else:
            print("\nDropping highly correlated columns (correlation > 0.95):")            
            for drop_col, keep_col, corr in high_corr_pairs:
                df = df.drop(columns=drop_col)
                print(f"  - '{drop_col}' is {corr:.2f} correlated with '{keep_col}': dropped '{drop_col}'.")

        # Drop irrelevant columns
        df = df.drop(columns=self.irrelevant_columns)

        print(f"\nDropped {len(self.irrelevant_columns)} irrelevant columns: {self.irrelevant_columns}")

        # Drop deprecated columns
        df = df.drop(columns=self.deprecated_columns)

        print(f"\nDropped {len(self.deprecated_columns)} deprecated columns: {self.deprecated_columns}")

        # Drop columns where missing values exceed 75% (excluding identified relevant columns)
        # *Adjusted from 80% to 75% because there were columns with a number of missing values really close to 80%
        missing_values = df.isnull().mean()
        columns_to_drop = missing_values[missing_values > 0.75].index.tolist()
        columns_to_drop = [col for col in columns_to_drop if col not in self.relevant_columns]
        df = df.drop(columns=columns_to_drop)

        print(f"\nDropped {len(columns_to_drop)} columns with >75% missing values: {columns_to_drop}")

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

        print("\nExtracted 'DAY', 'MONTH', 'WEEKDAY' from 'DATE'")

        print("\nAfter Cleaning")
        print("Shape:", df.shape)
        print("Remaining Columns:", df.columns.tolist())

        return df
