"""
PART 1
by Rafael Carlo Posadas
"""

import numpy as np
import pandas as pd

df0 = pd.read_csv('KSI_dataset.csv')

""" 
========================= DATA EXPLORATION ========================== 
"""
# Display dataset shape, column names and dtypes.
print(f"Shape: {df0.shape} \n")
print(f"Columns: \n{df0.columns}\n")
print(f"Data types:")
print(f"{df0.info()}\n")

""" 
========================= DATA PRE-PROCESSING ========================= 
"""
# Duplicate original dataset
df1 = df0.copy()

# Identify missing values.
df1_null_count = df1.isnull().sum()
print(f"Missing values:\n{df1_null_count[df1_null_count > 0]}\n")

# Replace null values in 'Yes'/'No' columns with "No".
for col in df1.columns:
    if df1[col].isin(["Yes", None]).any():
        df1[col] = df1[col].fillna("No")

""" ===== removing insignificant columns ===== """
# Convert the DATE column to datetime. This will be dropped. Then extract day, month, and year into their own columns.
df1['DATE'] = pd.to_datetime(df1['DATE'])
df1['DAY_OF_WEEK'] = df1['DATE'].dt.day_name()
df1['DAY'] = df1['DATE'].dt.day
df1['MONTH'] = df1['DATE'].dt.month
df1['YEAR'] = df1['DATE'].dt.year
# Extract the hour component from TIME. This is not instructed, but might be useful for plot and model building.
df1['HOUR'] = (df1['TIME'] // 100).astype(int)

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

# Verify the shape of the updated dataset.
print(f"Shape after dropping columns: {df1.shape}\n")

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


""" ===== removing insignificant rows ===== """
# Identify columns with missing values greater than 0% but less than 3%.
missing_percentage = df1.isnull().sum() / len(df1) * 100
columns_to_keep = missing_percentage[(missing_percentage > 0) & (missing_percentage < 3)].index
print(f"Columns with missing values >0% and <3%:\n{columns_to_keep.tolist()}\n")

# Drop rows with missing values in those columns.
df1 = df1.dropna(subset=columns_to_keep)

# Remove duplicate rows according to 'ACCNUM' values
df1 = df1.drop_duplicates(subset='ACCNUM', keep='first')

# Verify the shape of the updated dataset after dropping rows.
print(f"Shape after dropping rows: {df1.shape}\n")
""" 
============================= STATISTICS ============================= 
"""
# I decided to do stats after pre-processing so that I can account
# for the separate date items and the completed binary columns.

# Describe the numerical & object type columns.
# print(df1.describe())
# print(f"Object descriptions: \n{df1.describe(include=['object'])}\n")

# Get ranges for numeric columns.
print(f"Numerical ranges:")
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