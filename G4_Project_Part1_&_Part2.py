"""
PART 1
by Rafael Carlo Posadas
"""

import numpy as np
import pandas as pd

df0 = pd.read_csv('C:/Users/maiko/Downloads/KSI_dataset.csv')

""" 
========================= DATA EXPLORATION ========================== 
"""
# Display dataset shape, column names and dtypes.
print(f"Shape: {df0.shape} \n")
print(f"Columns: \n{df0.columns}\n")
print(f"Data types: {df0.info()}\n")

""" 
========================= DATA PRE-PROCESSING ========================= 
"""

# Duplicate original dataset
df1 = df0.copy()

# Identify missing values.
df1_null_count = df1.isnull().sum()
print(f"Missing values:\n{df1_null_count[df1_null_count > 0]}\n")

'''Step 1: 
    Changing all empyt values in 'Yes'/'No' columns with "No"'''

# Replace null values in 'Yes'/'No' columns with "No".
# Columns Pedestrian ==> diability 
for col in df1.columns:
    if df1[col].isin(["Yes", None]).any():
        df1[col] = df1[col].fillna("No")
        
#Let's check the shape of our columns, and how many null value remains
print(f"Shape: {df1.shape} \n")
print(f"null values: {df1.isnull().sum()}")


'''step 2: 
    Extract and Drop the "DATE" column '''

# Convert the DATE column to datetime. This will be dropped. Then extract day, month, and year into their own columns.
df1['DATE'] = pd.to_datetime(df1['DATE'])
df1['DAY_OF_WEEK'] = df1['DATE'].dt.day_name()
df1['DAY'] = df1['DATE'].dt.day
df1['MONTH'] = df1['DATE'].dt.month
df1['YEAR'] = df1['DATE'].dt.year
# Extract the hour component from TIME. This is not instructed, but might be useful for plot and model building.
df1['HOUR'] = (df1['TIME'] // 100).astype(int)

#Let's check the shape of our columns, and how many null value remains
print(f"Shape: {df1.shape} \n")
print(f"null values: {df1.isnull().sum()}")


'''step 3(Added part):
    Droping all the duplicated value(by rows) before dropping any columns based on the "ACCNUM" column '''

#ACCNUM column has many douplicated numbers, and 
#this indicates that we have lots of repeated rows
#So, based on the ACCNUM column, we'll drop the entire row

# Drop all duplicates based on the ACCNUM column
df1 = df1.drop_duplicates(subset=['ACCNUM'])

#Let's check the shape of our columns, and how many null value remains
print(f"Shape: {df1.shape} \n")
print(f"null values: {df1.isnull().sum()}")


'''Step 4: 
    Drop columns with very high missing values(columns which miss > 80% of their values)
    and irrelevant columns'''

# Identify columns with over 80% missing values. These will be dropped.
missing_percentage = df1_null_count / len(df1) * 100
# print(f"Percentage of missing values: \n{missing_percentage}\n")
columns_to_drop = missing_percentage[missing_percentage > 80].index.tolist()

# Drop the columns with over 80% missing values.
# The added columns are deemed redundant and instructed to be dropped.
# 'WARDNUM' and 'POLICE_DIVISION' were not found in the table, so they're excluded.
columns_to_drop.extend([
    'DATE',
    'OBJECTID',
    'INDEX',
    'NEIGHBOURHOOD_158',
    'NEIGHBOURHOOD_140',
    'x',
    'y',
    'INITDIR',
    'STREET1',
    'STREET2',
    'DIVISION',
    # Below are my suggestions based on PSDP_Open_Data_Documentation description.
    'ACCNUM', # Accident Number
    'OFFSET', # Distance and direction of the Collision
    ])
print(f"Columns to be removed:\n{columns_to_drop}\n")
df1 = df1.drop(columns=columns_to_drop)

#Let's check the shape of our columns, and how many null value remains
print(f"Shape: {df1.shape} \n")
print(f"null values: {df1.isnull().sum()}")


'''Step 5: 
    Modify the target column'''
""" 
'ACCLASS' is identified to be the target column as this classifies 
the accident type - according to the PSDP_Open_Data_Documentation -
and also due to its values being only 'Fatal' or 'Non-Fatal'.
"""
# Identify all values in ACCLASS column for correction.
print("'ACCLASS' before correction:")
print(f"unique: {df1['ACCLASS'].unique().tolist()}")
print(f"nulls : {df1['ACCLASS'].isnull().sum()}")

""" 
Only 'Property Damage' was instructed to be corrected, but it has one
null value that seems insignificant enough to qualify for correction.
"""
# Change incorrect values to "Non-Fatal".
# df1['ACCLASS'] = df1['ACCLASS'].replace(['Non-Fatal Injury', 'Property Damage O', None], 'Non-Fatal')
df1['ACCLASS'] = df1['ACCLASS'].apply(lambda x: 'Non-Fatal' if x != 'Fatal' else x)

# Verify updated ACCLASS values.
print("'ACCLASS' after correction:") 
print(f"unique: {df1['ACCLASS'].unique().tolist()}")
print(f"nulls : {df1['ACCLASS'].isnull().sum()}")

#Let's check the shape of our columns, and how many null value remains
print(f"Shape: {df1.shape} \n")
print(f"null values: {df1.isnull().sum()}")


'''Step 6: 
    Removing insignificant rows based on the data of columns with higher amount of data '''
    
# Identify columns with missing values greater than 0% but less than 3%.
missing_percentage = df1.isnull().sum() / len(df1) * 100
columns_to_keep = missing_percentage[(missing_percentage > 0) & (missing_percentage < 3)].index
print(f"Columns with missing values >0% and <3%:\n{columns_to_keep.tolist()}\n")

# Drop rows with missing values in those columns.
df1 = df1.dropna(subset=columns_to_keep)

#Let's check the shape of our columns, and how many null value remains
print(f"Shape: {df1.shape} \n")
print(f"null values: {df1.isnull().sum()}")


#Now evry column doesn't have a null values except the
#VEHTYPE column which have 177 null values. As its categorical column lets check the distribution


'''Step 7:
    Modify the null values remaining in the column "VEHTYPE"  '''


#Inorder to know how to choose the best category for the null value, first
# Check the distribution of the categories in the column 
category_counts = df1['VEHTYPE'].value_counts()
print(category_counts)

#And then check how the "VEHTYPE" column relate to 
#our target column "ACCLASS" using different visualization

import seaborn as sns
import matplotlib.pyplot as plt
# Create a crosstab between VEHTYPE and ACLASS
heatmap_data = pd.crosstab(df1['VEHTYPE'], df1['ACCLASS'])
crosstab = pd.crosstab(df1['VEHTYPE'], df1['ACCLASS'])

plt.figure(figsize=(12, 6))
sns.countplot(x='VEHTYPE', hue='ACCLASS', data=df1)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title('Distribution of VEHTYPE across ACLASS')
plt.xlabel('VEHTYPE')
plt.ylabel('Count')
plt.show()

crosstab.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('ACCLASS Distribution Across VEHTYPE')
plt.xlabel('VEHTYPE')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Heatmap of VEHTYPE vs ACLASS')
plt.xlabel('ACCLASS')
plt.ylabel('VEHTYPE')
plt.show()

# We have two dominant categories, 
#Automobile, Station Wagon        2550    
#Other                            1367


# So in conclusion, the  "Automobile, Station Wagon" is the most 
#frequent category, even though it's strong bias towards Non-Fetal. So, I decide to fill the 
#null values using the  "Automobile, Station Wagon" category. 

most_frequent_category = df1['VEHTYPE'].mode()[0]
df1['VEHTYPE'].fillna(most_frequent_category, inplace=True)

#Let's check the shape of our columns, and how many null value remains
print(f"Shape: {df1.shape} \n")
print(f"null values: {df1.isnull().sum()}")
df1.columns

# At this point we dont have any missing values in our dataset
""" 
============================= STATISTICS ============================= 
"""
# I decided to do stats after pre-processing so that I can account
# for the separate date items and the completed binary columns.

# Describe the numerical & object type columns.
# print(df1.describe())
# print(f"Object descriptions: \n{df1.describe(include=['object'])}\n")

# Get ranges for numeric columns.
print("Numerical ranges:")
for col in df1.columns:
    if df1[col].dtype not in ['object']:
        print(f"{col}: {df1[col].min()} to {df1[col].max()}")

# Calculate mean, median, and mode.
print(f"\nMean: \n{df1.mean(numeric_only=True)}\n")
print(f"Median: \n{df1.median(numeric_only=True)}\n")
print(f"Mode: \n{df1.mode().iloc[0]}\n")


""" 
================================ PLOTS ================================ 
"""
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Some macros for creating the graphs, plots and heatmap.
def stacked_hist(df, column, title, xlabel, ylabel, legend_labels=['Fatal', 'Non-Fatal']):
    """
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column to plot against ACCLASS.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    colors (list): Colors for the stacked bars.
    legend_labels (list): Labels for the legend.
    """
    # 'DAY_OF_WEEK' displays in alphabetical order by default, so it needs to be reorganized.
    if column == 'DAY_OF_WEEK':
        days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        df[column] = pd.Categorical(df[column], categories=days_order, ordered=True)
        df = df.sort_values(column)

    crosstab = pd.crosstab(df[column], df['ACCLASS'])
    ax = crosstab.plot(kind='bar', stacked=True, figsize=(12, 4))

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{int(height)}',
                    (x + width / 2, y + height),
                    ha='center', va='bottom',
                    fontsize=10, color='black', fontweight='bold')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(title='Accident Type', labels=legend_labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()

def horizontal_bar_graph(df, column, title, xlabel, ylabel, figsize=(12, 8)):
    """
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column to plot value counts for.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    figsize (tuple): The size of the figure (width, height).
    """
    value_counts = df[column].value_counts()
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=value_counts.values, y=value_counts.index)
    for i, value in enumerate(value_counts.values):
        ax.text(value + 5, i, str(value), va='center', fontweight='bold')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.show()

def map_accidents(df, accident_col='ACCLASS', lon_col='LONGITUDE', lat_col='LATITUDE', grid_resolution=100, density_threshold=0.5, figsize=(18, 13)):
    """
    Generates two plots:
    1. A scatter plot of accidents with a density-driven circle.
    2. A heatmap of accident density.

    df (pd.DataFrame): The dataframe containing accident data.
    accident_col (str): The column name for accident type.
    lon_col (str): The column name for longitude.
    lat_col (str): The column name for latitude.
    grid_resolution (int): Resolution for the heatmap grid (higher = smoother).
    density_threshold (float): Density threshold for the circle (0.5 for 50%).
    """
    # Plot 1: Scatter plot of all accidents with high-density circle
    plt.figure(figsize=figsize)
    non_fatal = df[df[accident_col] == 'Non-Fatal']
    fatal = df[df[accident_col] == 'Fatal']

    plt.scatter(non_fatal[lon_col], non_fatal[lat_col], alpha=0.5, color='blue', s=5, label='Non-Fatal')
    plt.scatter(fatal[lon_col], fatal[lat_col], alpha=0.5, color='red', s=5, label='Fatal')
    coordinates = np.vstack([df[lon_col], df[lat_col]])
    kde = gaussian_kde(coordinates)
    density = kde(coordinates)
    max_density_idx = np.argmax(density)
    max_density_longitude = df[lon_col].iloc[max_density_idx]
    max_density_latitude = df[lat_col].iloc[max_density_idx]
    threshold_density = np.max(density) * density_threshold
    distances = np.sqrt((df[lon_col] - max_density_longitude)**2 + (df[lat_col] - max_density_latitude)**2)
    radius = np.min(distances[density < threshold_density])

    circle = plt.Circle((max_density_longitude, max_density_latitude), radius=radius, color='yellow', fill=False, linestyle='-', linewidth=2, label=f'Top {density_threshold*100:.0f}% Density')
    plt.gca().add_patch(circle)
    # print(f"Accident hotspot center: ({max_density_latitude}, {max_density_longitude})")

    plt.title('Location of Accidents with Density-Driven Circle', fontsize=16)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.2, color='black')
    plt.legend(title='Legend', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot 2: Accident density heatmap.
    plt.figure(figsize=figsize)
    grid_x, grid_y = np.mgrid[
        df[lon_col].min():df[lon_col].max():grid_resolution * 1j,
        df[lat_col].min():df[lat_col].max():grid_resolution * 1j
    ]
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    grid_z = kde(grid_coords).reshape(grid_x.shape)

    plt.pcolormesh(grid_x, grid_y, grid_z, shading='auto', cmap='viridis')
    plt.scatter(df[lon_col], df[lat_col], color='white', alpha=0.25, s=5, label='Accidents')
    plt.title('Accident Density Heatmap', fontsize=16)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.grid(True, linestyle='--', alpha=1, color='white')
    plt.legend(title='Legend', fontsize=12)
    plt.tight_layout()
    plt.show()

# Plot 1-2: Distribution of accident types and severity.
horizontal_bar_graph(df1, 'ACCLASS', 'Distribution of Accidents (ACCLASS)', 'Number of Accidents', 'ACCLASS', figsize=(12, 2))
horizontal_bar_graph(df1, 'INJURY', 'Accidents by Injury Severity (INJURY)', 'Number of Accidents', 'INJURY', figsize=(12, 2))

# Plot 3-6: Accidents by year, month, day of week, and time of day.
stacked_hist(df1, 'YEAR', 'Accidents by Year', 'Year', 'Count')
stacked_hist(df1, 'MONTH', 'Accidents by Month', 'Month', 'Count')
stacked_hist(df1, 'DAY_OF_WEEK', 'Accidents by Day of the Week', 'Day of the Week', 'Count')
stacked_hist(df1, 'HOUR', 'Accidents by Time of Day', 'Hour (24h)', 'Count')

# Plot 7-10: Accidents by vehicle type, location, and road & light conditions.
horizontal_bar_graph(df1, 'VEHTYPE', 'Accidents by Vehicle Type (VEHTYPE)', 'Number of Accidents', 'VEHTYPE')
horizontal_bar_graph(df1, 'ACCLOC', 'Accidents by Location (ACCLOC)', 'Number of Accidents', 'ACCLOC', figsize=(12, 4))
horizontal_bar_graph(df1, 'RDSFCOND', 'Accidents by Road Surface Condition (RDSFCOND)', 'Number of Accidents', 'RDSFCOND', figsize=(12, 4))
horizontal_bar_graph(df1, 'LIGHT', 'Accidents by Light Condition (LIGHT)', 'Number of Accidents', 'LIGHT', figsize=(12, 4))

# Plot 11: Accidents by age of involved party.
def extract_numeric(age_label):
    if age_label == 'unknown':
        return -1  # Place 'unknown' first.
    elif age_label == 'Over 95':
        return 95  # Place 'Over 95' last.
    else:
        return int(age_label.split(" ")[0])
sorted_labels = sorted(df1['INVAGE'].unique(), key=extract_numeric)
plt.figure(figsize=(18, 5))
ax = sns.countplot(data=df1, x='INVAGE', order=sorted_labels, hue='ACCLASS')
for p in ax.patches:
    height = p.get_height()
    if height != 0:  # Added this to remove the random '0' in the first bar.
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')
plt.title("Distribution of ages involved in accidents (INVAGE)", fontsize=16)
plt.xlabel("Age Group", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()

# Plot 12: Accidents by district
horizontal_bar_graph(df1, 'DISTRICT', 'Accidents by Municipal Region (DISTRICT)', 'Number of Accidents', 'DISTRICT', figsize=(12, 2))

# Plot 13-14: Geographical location of accidents.
# It's one command but this will display 2 maps.
map_accidents(df1, accident_col='ACCLASS', lon_col='LONGITUDE', lat_col='LATITUDE', grid_resolution=500, density_threshold=0.5, figsize=(18, 13))


"""         DELETE THIS BEFORE SUBMISSION
Insights: 
- Dataset presents a heavy bias on 'non-fatal' accidents. That's good in the safety sense, but not as good for 
statistical analysis as there is fewer data points on fatal accidents.
- Total accidents on a yearly basis are declining, that's good news, and fatalities tend to be on a very slow decline 
but is less predictable.
- Most accidents happen during the summer when people are on school break (JUN-AUG) and possibly temporarily high 
population from tourism.
- Fridays are the most prevalent in total and non-fatal accidents; it's the weekend and people tend to go crazy.
- Station wagons and 'other' vehicles account for more than half all the accidents.
- Dry weather allows for faster (aggressive) driving, thus leads in accidents by road surface condition. While wet 
conditions inhibits braking power and sometimes vehicles can experience hydroplaning.
- Ages 20-54 are the most involved in these accidents, while the age group 20-24 have the highest among all accounts. 
This is when many new drivers in Ontario receive their full G license, and are just starting to drive on their own with 
less supervision and driving restrictions. Number of accidents decrease as age increases, this is likely due to the 
elderly staying at home more and also driving less.
- Most accidents happen during daylight, just a bit more than in dark lighting conditions, probably due to pedestrian 
higher activity during the day and many people would already be home after their 9am-5pm jobs.
- In terms of density, downtown Toronto seems to have the highest concentration of accidents. This is anticipated as 
it is the densest by residential and general pedestrian population out of all the other regions. Also, the downtown 
area has never been very compatible with cars (old roads dedicated to trams, pedestrian walking and cycling). It has 
become plagued with heavy congestion from personal vehicles and endless construction. Because of these, more pedestrians, 
cyclists and drivers are active at any time and much closer to each other, which could increase the chances of accidents.
- While these plots don't necessarily help with selecting features to build the classifier model, they can provide 
context for the human when interpreting testing results.
"""

""" 


########################################### PART 2: Data Modeling


=================== Convert Catagorical Columns to Numeric columns ================================ 
"""
#Check how the dataset looks and how much null values we have
print(f"Shape: {df1.shape} \n")
print(f"null values: {df1.isnull().sum()}")
df1.columns

'''Step 1: 
    The two columns 'HOOD_158' and 'HOOD_140' which have both numeric and "NSA" values'''
    
# First let's see how does the categorical columns looks like 
# Till this point we have 22 columns, and only 8 of them are numeric columns
# Get the number of unique categories for each categorical column
categorical_columns = df1.select_dtypes(include=['object']).columns
# Print number of unique categories for each categorical column
for col in categorical_columns:
    print(f"{col}: {df1[col].nunique()} unique categories")
    

#The 'HOOD_158' and 'HOOD_140' columns have combination of numeric and 'NSA' values 
#I have to remove or replace them with a number like median, but they seems like address and 
#it is not effective to use a median, insted i decide to remove the rows which have 'NSA'

# the rows that have 'NSA' in either 'HOOD_158' or 'HOOD_140'
rows_to_drop = df1[df1['HOOD_158'].isin(['NSA']) | df1['HOOD_140'].isin(['NSA'])]

# Print the number of rows that will be dropped
print(f"Number of rows to drop: {len(rows_to_drop)}")

print(f"Shape befor droping 'NSA' values: {df1.shape} \n")
# Drop rows by index
df1 = df1.drop(rows_to_drop.index)

# Check 
print(f"Shape after droping 'NSA' values: {df1.shape} \n")

#The columns are treated as an objest, even after remving all the 'NSA' values, So 
#Just try to conver them to numeric value by using pd.to_numeric() and they will exclude in 
#the encoding which comes after

# Convert HOOD_158 and HOOD_140 to numeric
df1['HOOD_158'] = pd.to_numeric(df1['HOOD_158'])
df1['HOOD_140'] = pd.to_numeric(df1['HOOD_140'])

# Check the column types again to ensure they've been converted to numeric
print(df1.dtypes)

# Re-run the categorical columns check
categorical_columns = df1.select_dtypes(include=['object']).columns
# Print number of unique categories for each categorical column
for col in categorical_columns:
    print(f"{col}: {df1[col].nunique()} unique categories")


##########################

'''Step 2:
    Handling The remaining Categorical Columns - Data transformations'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline as imPipeline 


# Separate target (y) from features (X)
X = df1.drop('ACCLASS', axis=1)  # Drop the target column 'ACCLASS' from the feature set
y = df1['ACCLASS']  # The target column 

# Stratified Train-Test Split to maintain class balance in train/test sets
#stratify=y helps us to ensure that the class distribution is consistent 
#across the training and test datasets, making the model training and evaluation more reliable, 
#especially when dealing with imbalanced classes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=64)

#  Define numeric and categorical columns
numeric_columns = X.select_dtypes(include=['int32', 'int64', 'float64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns


''' ========== standardizations and OneHotEncoding ============== '''

# Define preprocessing for numerical columns (scaling + missing data handling)
numeric_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())  # Apply StandardScaler to numerical columns
])

#  Define preprocessing for categorical columns (one-hot encoding)
categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))  # Apply One-Hot Encoding (drop='first' to avoid multicollinearity)
])

#  Define a ColumnTransformer to apply the different pipelines to different columns
preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', numeric_pipeline, numeric_columns),  # Apply numeric pipeline to numerical columns
        ('cat', categorical_pipeline, categorical_columns)  # Apply categorical pipeline to categorical columns
    ]
)


#############################

'''Step 3:
    Building the Pipeline - includes 
    
    Handling Class Imbalance with SMOTE inside the pipeline
    
    Feature selection- # Feature Selection using Recursive Feature Elimination (RFE) - 
    #RFE recursively removes the least important features (based on the classifier) 
    #until we are left with the most important ones.
    '''

# Define the model
classifier = RandomForestClassifier(random_state=64)

# Define the feature selection
feature_selector = RFE(estimator=RandomForestClassifier(random_state=64), n_features_to_select=10)

# Define the SMOTE step
smote = SMOTE(sampling_strategy='auto', random_state=64)

# Build a complete pipeline for preprocessing and classification
# Build the pipeline with SMOTE integrated inside
pipeline = imPipeline(steps=[
    ('preprocessor', preprocessor),  # Apply preprocessing (scaling + encoding)
    ('smote', smote),  # Apply SMOTE for balancing classes
    ('feature_selector', feature_selector),  # Feature selection
    ('classifier', classifier)  # Model training
])


# Train the model using the pipeline on the training data
pipeline.fit(X_train, y_train)


###############################

'''Step 4:
    Evaluating the Model on Test Data'''


# Evaluate the model on the test data
y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))



"""
While i tried to analyze the report, i got the following key Insights from the first classification report:
    
Imbalanced Class Performance: The model is performing very well on the Non-Fatal class (with high precision, recall, and F1-score). 
However, the performance on the Fatal class is much worse, with low precision, recall, and F1-score.

Imbalanced Dataset: Since Non-Fatal incidents are more frequent than Fatal incidents (based on the support numbers), 
the model may be biased towards predicting Non-Fatal incidents. This is common in imbalanced datasets.

Low Recall for Fatal Class: The model is missing many actual Fatal incidents. Only 21% of the true Fatal 
cases are being identified (low recall).

Precision-Recall Tradeoff: Getting a high precision for Non-Fatal (0.89) and a relatively low precision for Fatal (0.37), 
which means that while the model is good at predicting Non-Fatal cases, 
it struggles with predicting Fatal cases accurately.
"""



'''####################################################'''


