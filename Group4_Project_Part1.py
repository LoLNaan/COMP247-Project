"""
PART 1
by Rafael Carlo Posadas
"""

import numpy as np
import pandas as pd

df0 = pd.read_csv('KSI_dataset.csv')

pd.set_option('display.max_columns', None)  # Show all columns.
# pd.set_option('display.max_rows', None)    # Show all rows.
pd.set_option('display.width', None)       # Auto-detect the display width.

print(""" ===================================== DATA EXPLORATION ====================================== """)
# Display dataset shape, column names, dtypes, and missing values.
print(f"Shape: {df0.shape} \n")
print(f"Columns: \n{df0.columns}\n")
print(f"Data types:")
print(f"{df0.info()}\n")
print(f"Missing values:\n{df0.isnull().sum()[df0.isnull().sum() > 0]}\n")


print(""" ===================================== DATA PRE-PROCESSING ===================================== """)
# Duplicate original dataset and identify missing values.
df1 = df0.copy()
df1_null_count = df1.isnull().sum()

# Impute the binary columns.
for col in df1.columns:
    if df1[col].isin(["Yes", None]).any():
        df1[col] = df1[col].fillna("No")


""" ===== Breakdown DATE column into smaller components ===== """
# Convert the DATE column to datetime. This will be dropped later.
df1['DATE'] = pd.to_datetime(df1['DATE'])
# Extract day, month, and year into their own columns.
df1['DAY_OF_WEEK'] = df1['DATE'].dt.day_name()
df1['DAY'] = df1['DATE'].dt.day
df1['MONTH'] = df1['DATE'].dt.month
df1['YEAR'] = df1['DATE'].dt.year
# Extract the hour component. This is not instructed, just personal preference for plot usage.
df1['HOUR'] = (df1['TIME'] // 100).astype(int)

" ==== Remove insignificant columns (>80% missing values) ==== "
# Identify columns with over 80% missing values.
missing_percentage = df1.isnull().sum() / len(df1) * 100
print(f"Missing values percentages: \n{missing_percentage[missing_percentage > 0]}\n")
columns_to_drop = missing_percentage[missing_percentage >= 80].index.tolist()
print(f"Columns with >80% missing values:\n{columns_to_drop}\n")

# - Drop columns with over 80% missing values.
# - The appended columns are instructed as redundant and to be dropped.
# - 'WARDNUM' and 'POLICE_DIVISION' were not found in the table, so they're excluded.
# - 'OFFSET' has 79.8% is missing; close enough to 80% so it's also being dropped.
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
    # Manual addition:
    'OFFSET'
    ])
df1 = df1.drop(columns=columns_to_drop)

# Verify the shape of the updated dataset.
print(f"Shape after dropping columns: {df1.shape}\n")

""" 
'ACCLASS' will be the target column as this classifies the accident type - according to the 
PSDP_Open_Data_Documentation - and also due to its values being only 'Fatal' or 'Non-Fatal'.
"""
# Identify all values in ACCLASS column for correction.
print("'ACCLASS' before correction:")
print(f"unique: {df1['ACCLASS'].unique().tolist()}")
print(f"nulls : {df1['ACCLASS'].isnull().sum()}\n")

# - Change incorrect values to "Non-Fatal".
# - Only 'Property Damage' was instructed to be corrected, but any other
#   value that's not 'Fatal' will be converted to 'Non-Fatal'.
df1['ACCLASS'] = df1['ACCLASS'].apply(lambda x: 'Non-Fatal' if x != 'Fatal' else x)

# Verify updated ACCLASS values.
print("'ACCLASS' after correction:")
print(f"unique: {df1['ACCLASS'].unique().tolist()}")
print(f"nulls : {df1['ACCLASS'].isnull().sum()}\n")


""" ===== Remove insignificant rows ===== """
# Identify columns with missing values greater than 0% but less than 3%.
missing_percentage = df1.isnull().sum() / len(df1) * 100
columns_to_keep = missing_percentage[(missing_percentage > 0) & (missing_percentage < 3)].index
print(f"Columns with missing values >0% and <3%:\n{columns_to_keep.tolist()}\n")

# Drop rows with missing values in those columns.
df1 = df1.dropna(subset=columns_to_keep)

# Verify the shape of the updated dataset after dropping rows.
print(f"Shape after dropping rows: {df1.shape}\n")


""" ==== Fix ACCNUM column ===="""
# Show any remaining columns with null values.
print(f"Remaining columns with nulls: \n{df1.isnull().sum()[df1.isnull().sum() > 0]}\n")

# - df1 is duplicated, df2 will be used for statistics and plotting.
# - Rows with duplicate values in ACCNUM are removed from df2. Nulls are kept for deeper cleaning.
# - Hash the rest of the dataset.
# - Check for duplicate hashes and drop them.
df2 = df1.copy()
df2 = df2[~df2['ACCNUM'].duplicated(keep='first') | df2['ACCNUM'].isna()]
df2['hash'] = df2.drop(columns=['ACCNUM']).apply(lambda row: hash(tuple(row)), axis=1)
print(f"Shape after dropping ACCNUM duplicates: {df2.shape}\n")
df2 = df2.drop_duplicates(subset='hash', keep='first')

# ACCNUM has effectively become a unique ID column, so it's no longer
# useful and will be dropped with the temporary 'hash' column.
df2 = df2.drop(columns=['hash', 'ACCNUM'])

# Verify the shape of the updated dataset after altering ACCNUM.
print(f"Shape after dropping hashed row duplicates: {df2.shape}\n")


print(""" ========================================= STATISTICS ========================================= """)
# I decided to do stats after pre-processing so that I can account
# for the separate date items and the completed binary columns.

# Describe the numerical & object type columns.
print(f"Numeric columns: \n{df2.describe()}\n")
print(f"Object columns: \n{df2.describe(include=['object'])}\n")

# Get ranges for numeric columns.
print(f"Numerical ranges:")
for col in df2.columns:
    if df2[col].dtype not in ['object']:
        print(f"{col}: {df2[col].min()} to {df2[col].max()}")

# Calculate mean, median, and mode.
print(f"\nMean: \n{df2.mean(numeric_only=True)}\n")
print(f"Median: \n{df2.median(numeric_only=True)}\n")
print(f"Mode: \n{df2.mode().iloc[0]}\n")


print(""" =========================================== PLOTS =========================================== """)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

dpi = 300

# Some macros for creating the graphs, plots and heatmap.
def stacked_hist(df, column, title, xlabel, ylabel, figsize=(12, 4)):
    """
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column to plot against ACCLASS.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    colors (list): Colors for the stacked bars.
    legend_labels (list): Labels for the legend.
    """
    if column == 'DAY_OF_WEEK':  # If the column is 'DAY_OF_WEEK', reorder the days of the week.
        days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        df[column] = pd.Categorical(df[column], categories=days_order, ordered=True)
        df = df.sort_values(column)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    crosstab = pd.crosstab(df[column], df['ACCLASS'])
    crosstab.plot(kind='bar', stacked=True, ax=ax)

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
    ax.legend(title='Accident Type', labels=['Fatal', 'Non-Fatal'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()

def hbar(df, column, title, xlabel, ylabel, figsize=(12, 8)):
    """
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column to plot value counts for.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    figsize (tuple): The size of the figure (width, height).
    """
    value_counts = df[column].value_counts()
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.barplot(x=value_counts.values, y=value_counts.index)
    for i, value in enumerate(value_counts.values):
        ax.text(value + 5, i, str(value), va='center', fontweight='bold')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_age_distribution(df=df1, age_column='INVAGE', hue_column='ACCLASS', figsize=(18, 5)):
    def extract_numeric(age_label):
        if age_label == 'unknown':
            return -1  # Place 'unknown' first.
        elif age_label == 'Over 95':
            return 95  # Place 'Over 95' last.
        else:
            return int(age_label.split(" ")[0])

    sorted_labels = sorted(df[age_column].unique(), key=extract_numeric)
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.countplot(data=df, x=age_column, order=sorted_labels, hue=hue_column)
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

def map_accidents(df, grid_resolution=100, density_threshold=0.5, dot_size=5, figsize=(18, 13)):
    """
    Generates two plots:
    1. A scatter plot of accidents with a density-driven circle.
    2. A heatmap of accident density.

    df (pd.DataFrame): The dataframe containing accident data.
    accident_col (str): The column name for accident type (e.g., 'ACCLASS').
    lon_col (str): The column name for longitude (e.g., 'LONGITUDE').
    lat_col (str): The column name for latitude (e.g., 'LATITUDE').
    grid_resolution (int): Resolution for the heatmap grid (higher = smoother).
    density_threshold (float): Density threshold for the circle (e.g., 0.5 for 50%).
    """
    accident_col='ACCLASS'
    lon_col='LONGITUDE'
    lat_col='LATITUDE'
    markerscale = 5
    grid_line_alpha = 0.5
    plot_alpha = 0.5
    title_fontsize = 16
    axis_fontsize = 14
    legend_fontsize = 12

    # Plot 1: Scatter plot of all accidents with high-density circle
    plt.figure(figsize=figsize, dpi=dpi)
    non_fatal = df[df[accident_col] == 'Non-Fatal']
    fatal = df[df[accident_col] == 'Fatal']

    plt.scatter(non_fatal[lon_col], non_fatal[lat_col], alpha=plot_alpha, color='blue', s=dot_size, label='Non-Fatal')
    plt.scatter(fatal[lon_col], fatal[lat_col], alpha=0.5, color='red', s=dot_size, label='Fatal')

    coordinates = np.vstack([df[lon_col], df[lat_col]])
    kde = gaussian_kde(coordinates)
    density = kde(coordinates)
    max_density_idx = np.argmax(density)
    max_density_longitude = df[lon_col].iloc[max_density_idx]
    max_density_latitude = df[lat_col].iloc[max_density_idx]
    threshold_density = np.max(density) * density_threshold
    distances = np.sqrt((df[lon_col] - max_density_longitude)**2 + (df[lat_col] - max_density_latitude)**2)
    radius = np.min(distances[density < threshold_density])

    circle = plt.Circle((max_density_longitude, max_density_latitude),
                        radius=radius, color='yellow', fill=False, linestyle='-', linewidth=2,
                        label=f'Top {density_threshold*100:.0f}% Density')
    plt.gca().add_patch(circle)
    print(f"Hotspot center: ({max_density_longitude} LONG, {max_density_latitude} LAT)")

    plt.title('Location of Accidents', fontsize=title_fontsize)
    plt.xlabel(lon_col, fontsize=axis_fontsize)
    plt.ylabel(lat_col, fontsize=axis_fontsize)
    plt.grid(True, linestyle='--', alpha=grid_line_alpha/2, color='black')
    plt.legend(title='Legend',
               fontsize=legend_fontsize,
               markerscale=markerscale,
               facecolor='white',
               loc='lower right',
               )
    plt.tight_layout()
    plt.show()

    # Plot 2: Accident density heatmap.
    plt.figure(figsize=figsize, dpi=dpi)
    grid_x, grid_y = np.mgrid[
        df[lon_col].min():df[lon_col].max():grid_resolution * 1j,
        df[lat_col].min():df[lat_col].max():grid_resolution * 1j
    ]
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    grid_z = kde(grid_coords).reshape(grid_x.shape)

    plt.pcolormesh(grid_x, grid_y, grid_z, shading='auto', cmap='magma')
    plt.scatter(df[lon_col], df[lat_col], color='white', alpha=plot_alpha, s=dot_size/2, label='Accidents')
    plt.title('Accident Density Heatmap', fontsize=title_fontsize)
    plt.xlabel(lon_col, fontsize=axis_fontsize)
    plt.ylabel(lat_col, fontsize=axis_fontsize)
    plt.grid(True, linestyle='--', alpha=grid_line_alpha/5, color='white')
    plt.legend(title='Legend',
               framealpha=1,
               fontsize=legend_fontsize,
               markerscale=markerscale,
               facecolor='grey',
               loc='lower right',
               )
    plt.tight_layout()
    plt.show()

# Plot 1-2: Distribution of accident types and severity.
hbar(df2, 'ACCLASS', 'Distribution of Accidents (ACCLASS)', 'Number of Accidents', 'ACCLASS', figsize=(12, 2))
hbar(df2, 'INJURY', 'Accidents by Injury Severity (INJURY)', 'Number of Accidents', 'INJURY', figsize=(12, 2))

# Plot 3-6: Accidents by year, month, day of week, and time of day.
stacked_hist(df1, 'YEAR', 'Accidents by YEAR (ACCNUM unaltered)', 'YEAR', 'Count')
stacked_hist(df2, 'YEAR', 'Accidents by YEAR (ACCNUM corrected)', 'YEAR', 'Count')
stacked_hist(df1, 'MONTH', 'Accidents by MONTH', 'MONTH', 'Count')
stacked_hist(df1, 'DAY_OF_WEEK', 'Accidents by Day of the Week', 'DAY_OF_WEEK', 'Count')
stacked_hist(df1, 'HOUR', 'Accidents by Time of Day', 'HOUR (24h)', 'Count')

# Plot 7-10: Accidents by vehicle type, location, and road & light conditions.
hbar(df2, 'VEHTYPE', 'Accidents by Vehicle Type (VEHTYPE)', 'Number of Accidents', 'VEHTYPE')
hbar(df2, 'ACCLOC', 'Accidents by Location (ACCLOC)', 'Number of Accidents', 'ACCLOC', figsize=(12, 4))
hbar(df2, 'RDSFCOND', 'Accidents by Road Surface Condition (RDSFCOND)', 'Number of Accidents', 'RDSFCOND', figsize=(12, 4))
hbar(df2, 'LIGHT', 'Accidents by Light Condition (LIGHT)', 'Number of Accidents', 'LIGHT', figsize=(12, 4))

# Plot 11-12: Accidents by involved parties.
plot_age_distribution(df=df2, figsize=(16, 4))
hbar(df2, 'DISTRICT', 'Accidents by Municipal Region (DISTRICT)', 'Number of Accidents', 'DISTRICT', figsize=(12, 2))

# This single command will output 2 maps.
map_accidents(df2, grid_resolution=500, density_threshold=0.5, dot_size=2, figsize=(18, 13))

"""
Insights: 
- Dataset presents a heavy bias on 'non-fatal' accidents. That's good in the safety sense, but not as good for 
statistical analysis as there is fewer data points to analyze fatal accidents.
- The histogram for yearly accidents has a questionable discrepancy. When analyzing the dataset where ACCNUM duplicates
were not yet removed, the total accidents on a yearly basis are shown to be steadily declining, that's good news. Yearly
fatalities tend to also be on a very slow decline but is less predictable. Then, when duplicates are removed, all years 
shrunk proportionally except for years 2015-2019 as their accident count barely shrunk. I'm not sure why this is.
- Most accidents happen during the summer months when people are on school break (JUN-AUG) and when the population is 
temporarily higher due to vacationers and tourists.
- Fridays are the most prevalent in total accidents; it's the weekend and more drivers leave and enter the city, also
people tend to be crazier on the road (personal experience).
- Station wagons and 'other' vehicles account for more than half all the accidents.
- Dry weather allows for faster (more aggressive) driving, thus leads in accidents by road surface condition. While wet 
conditions inhibits braking power and are more slippery. Sometimes vehicles can experience hydroplaning.
- Ages 20-54 are the most involved in these accidents, while the age group 20-24 have the highest among all accounts. 
This is when many new drivers in Ontario receive their full G license, and are just starting to drive on their own with 
less supervision and driving restrictions. Number of accidents decrease as age increases, this is likely due to the 
elderly staying at home more and also driving less.
- Most accidents happen during daylight, just a bit more than in dark lighting conditions, probably due to higher
pedestrian activity during the day and many people would already be home after their 9am-5pm jobs.
- In terms of density, downtown Toronto seems to have the highest concentration of accidents. This is anticipated as 
it is the densest by residential and general pedestrian population out of all the other regions. The downtown 
area has never been very compatible with cars (historic roads dedicated to trams, cycling & pedestrian walking). It has 
become plagued with heavy congestion by personal vehicles and endless construction. Because of these, more pedestrians, 
cyclists and drivers are active at any time and much closer to each other, which could increase the chances of accidents.
- Superficially, the graphs don't seem to indicate which features would best correlate to fatal accidents. But they can 
provide some insight for the human when interpreting prediction results.
"""