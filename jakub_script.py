# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_excel("Olympics_dataset_Final_product.xlsx")

# %%
print(df.info())
print(df.describe())

# %%
# Choose data from after 1924 to avoid different metric systems and weird sports
# Choose only summer games
df_1924 = df[df['year'] >= 1924]
df_1924_summer = df_1924[df_1924['season']=="Summer Olympics"]
df_swimming = df_1924_summer[df_1924_summer['sport'] == 'Swimming']

# %%
# Encode medals with values: Gold = 3, Silver = 2, Bronze = 1, None = 0
medal_to_points = {
    'Gold': 3,
    'Silver': 2,
    'Bronze': 1,    
    None: 0,
}

df_swimming['medal_points'] = df_swimming['medal'].apply(lambda x: medal_to_points.get(x, 0))

# %%
df_swimming['event'].unique()

# %%
import pandas as pd
import numpy as np
import re

# --- Normalization Function ---
def normalize_event_description(description_series):
    """Normalizes event descriptions, e.g., replacing '×' with 'x'."""
    if not isinstance(description_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    return description_series.str.replace('×', 'x', regex=False).astype(str) # Ensure string type

# --- Gender Extraction Function ---
def extract_gender(description_series):
    """Extracts gender (Men, Women, Mixed, Unknown) from a series of event descriptions."""
    if not isinstance(description_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    
    gender = description_series.str.extract(r'(Men|Women|Mixed)(?!.*\b(?:Men|Women|Mixed)\b)', flags=re.IGNORECASE)[0]
    # For "Relay Only Athlete" or if no gender is found, mark as 'Unknown'
    gender = gender.fillna('Unknown')
    # Handle cases like "Relay Only Athlete" specifically if they don't fit the regex
    gender.loc[description_series.str.contains('Relay Only Athlete', case=False, na=False)] = 'Unknown'
    return gender

# --- Event Type Extraction Function ---
def extract_event_type(description_series):
    """Extracts event type (Relay, Individual, Team, etc.) from a series of event descriptions."""
    if not isinstance(description_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    def determine_type(description):
        if pd.isna(description):
            return 'Unknown'
        description_lower = str(description).lower() # Ensure it's a string
        if 'relay' in description_lower:
            return 'Relay'
        if 'team swimming' in description_lower:
            return 'Team Swimming'
        if 'handicap' in description_lower:
            return 'Handicap'
        if 'individual medley' in description_lower:
            return 'Individual Medley'
        if description_lower == 'relay only athlete':
            return 'Athlete Role'
        if any(kw in description_lower for kw in ['plunge for distance', 'underwater swimming', 'obstacle course']):
            return 'Special Individual'
        if any(g in description_lower for g in ['men', 'women', 'mixed']):
             if not any(kw in description_lower for kw in ['relay', 'team', 'handicap']):
                return 'Individual'
        return 'Unknown'
    
    return description_series.apply(determine_type)

# --- Stroke Extraction Function ---
def extract_stroke(description_series):
    """Extracts the swimming stroke from a series of event descriptions."""
    if not isinstance(description_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    def determine_stroke(description):
        if pd.isna(description):
            return 'Unknown'
        description_lower = str(description).lower() # Ensure it's a string
        if 'freestyle' in description_lower:
            if 'for sailors' in description_lower:
                return 'Freestyle For Sailors'
            return 'Freestyle'
        if 'backstroke' in description_lower:
            return 'Backstroke'
        if 'breaststroke' in description_lower:
            return 'Breaststroke'
        if 'butterfly' in description_lower:
            return 'Butterfly'
        if 'medley' in description_lower: # Catches Individual Medley and Medley Relay
            return 'Medley'
        if 'obstacle course' in description_lower:
            return 'Obstacle Course'
        if 'underwater swimming' in description_lower:
            return 'Underwater Swimming'
        if 'plunge for distance' in description_lower:
            return 'Plunge For Distance'
        if 'team swimming' in description_lower:
            return 'Team Event' # Specific stroke often not mentioned for "Team Swimming"
        if description_lower == 'relay only athlete':
            return 'Not Applicable'
        return 'Unknown'
        
    return description_series.apply(determine_stroke)

# --- Distance and Unit Extraction Function ---
def extract_distance_info(description_series):
    """
    Extracts total distance, unit, relay legs, and leg distance 
    from a series of event descriptions.
    Returns a DataFrame with these columns.
    """
    if not isinstance(description_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    def parse_single_description(description):
        if pd.isna(description):
            return np.nan, 'Unknown', np.nan, np.nan
        
        description_str = str(description) # Ensure it's a string

        # Handle relay distances: "4 x 100 metres" or "4 x 50 yards"
        relay_match = re.search(r'(\d+)\s*x\s*([\d,]+)\s*(metres|yards|mile)', description_str, re.IGNORECASE)
        if relay_match:
            legs = int(relay_match.group(1))
            leg_distance_str = relay_match.group(2).replace(',', '')
            leg_distance = float(leg_distance_str)
            unit = relay_match.group(3).lower()
            total_distance = legs * leg_distance
            return total_distance, unit, float(legs), leg_distance

        # Handle individual distances: "1,200 metres", "440 yards", "1 mile"
        individual_match = re.search(r'([\d,]+(?:\.\d+)?)\s*(metres|yards|mile)', description_str, re.IGNORECASE)
        if individual_match:
            distance_str = individual_match.group(1).replace(',', '')
            distance = float(distance_str)
            unit = individual_match.group(2).lower()
            return distance, unit, 1.0, distance

        # Handle special cases
        desc_lower = description_str.lower()
        if 'plunge for distance' in desc_lower:
            return np.nan, 'Special (Plunge)', 1.0, np.nan
        if 'underwater swimming' in desc_lower:
            return np.nan, 'Special (Underwater)', 1.0, np.nan
        if 'obstacle course' in desc_lower: # Often has a distance, e.g., "200 metres Obstacle Course"
            # This case should ideally be caught by individual_match if distance is present.
            # If not, it implies distance is not specified in the standard format.
             obstacle_match = re.search(r'([\d,]+)\s*metres\s*Obstacle Course', description_str, re.IGNORECASE)
             if obstacle_match:
                 distance_str = obstacle_match.group(1).replace(',', '')
                 distance = float(distance_str)
                 return distance, 'metres', 1.0, distance
             return np.nan, 'Special (Obstacle)', 1.0, np.nan
        if 'relay only athlete' in desc_lower:
            return np.nan, 'Not Applicable', np.nan, np.nan
        
        return np.nan, 'Unknown', np.nan, np.nan

    results = description_series.apply(lambda x: pd.Series(parse_single_description(x)))
    results.columns = ['total_distance', 'unit', 'relay_legs', 'relay_leg_distance']
    return results

# --- Distance Conversion to Meters ---
def convert_distances_to_meters(distance_df):
    """
    Converts 'total_distance' and 'relay_leg_distance' to meters.
    Assumes distance_df has 'total_distance', 'relay_leg_distance', and 'unit' columns.
    """
    if not isinstance(distance_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not all(col in distance_df.columns for col in ['total_distance', 'relay_leg_distance', 'unit']):
        raise ValueError("DataFrame must contain 'total_distance', 'relay_leg_distance', and 'unit' columns.")

    df = distance_df.copy() # Work on a copy

    def convert_to_m(dist_val, unit_val):
        if pd.isna(dist_val) or pd.isna(unit_val) or not isinstance(unit_val, str):
            return np.nan
        
        unit_lower = unit_val.lower()
        if 'yards' in unit_lower:
            return dist_val * 0.9144
        if 'mile' in unit_lower:
            return dist_val * 1609.34
        if 'metres' in unit_lower or 'meters' in unit_lower : # Allow for 'meters' spelling
            return dist_val
        # For 'Special', 'Not Applicable', 'Unknown' units, distance in meters is undefined
        if any(spec_unit in unit_lower for spec_unit in ['special', 'not applicable', 'unknown']):
            return np.nan
        return np.nan # Default for unrecognized units

    df['total_distance_meters'] = df.apply(lambda row: convert_to_m(row['total_distance'], row['unit']), axis=1)
    df['relay_leg_distance_meters'] = df.apply(lambda row: convert_to_m(row['relay_leg_distance'], row['unit']), axis=1)
    
    return df[['total_distance_meters', 'relay_leg_distance_meters']]


# --- Main Function to Apply All Extractions ---
def extract_all_features(df, column_name):
    """
    Applies all extraction functions to the specified column of the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing event descriptions.
        
    Returns:
        pd.DataFrame: The DataFrame with new extracted feature columns.
    """
    if column_name not in df_swimming.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Create a working copy to avoid SettingWithCopyWarning
    df_processed = df_swimming.copy()

    # 1. Normalize descriptions
    df_processed['normalized_description'] = normalize_event_description(df_processed[column_name])
    
    # 2. Extract Gender
    df_processed['gender'] = extract_gender(df_processed['normalized_description'])
    
    # 3. Extract Event Type
    df_processed['event_type'] = extract_event_type(df_processed['normalized_description'])
    
    # 4. Extract Stroke
    df_processed['stroke'] = extract_stroke(df_processed['normalized_description'])
    
    # 5. Extract Distance Info (total_distance, unit, relay_legs, relay_leg_distance)
    distance_info_df = extract_distance_info(df_processed['normalized_description'])
    df_processed = pd.concat([df_processed, distance_info_df], axis=1)
    
    # 6. Convert distances to meters
    meter_distances_df = convert_distances_to_meters(df_processed[['total_distance', 'relay_leg_distance', 'unit']])
    df_processed = pd.concat([df_processed, meter_distances_df], axis=1)
    
    return df_processed

# --- Example Usage ---
if __name__ == '__main__':
    # Create a sample DataFrame (replace with your actual DataFrame)
    sample_data = {
        'event_details': [
            '1,200 metres Freestyle, Men', 
            '500 metres Freestyle, Men',
            '100 metres Freestyle For Sailors, Men',
            '4 × 50 yards Freestyle Relay, Men',
            'Plunge For Distance, Men',
            'Underwater Swimming, Men',
            '4 x 100 metres Medley Relay, Women',
            '200 metres Obstacle Course, Men',
            'Relay Only Athlete',
            '1 mile Freestyle, Men',
            'Unknown Event Type', # To test unknown handling
            np.nan # To test NaN handling
        ],
        'other_column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }
    my_dataframe = pd.DataFrame(sample_data)
    
    # Specify the column that contains the event descriptions
    event_description_column = 'event_details' 
    
    print("Original DataFrame:")
    print(my_dataframe)
    print("-" * 50)
    
    # Apply the feature extraction
    try:
        df_with_features = extract_all_features(my_dataframe, event_description_column)
        
        print("\nDataFrame with Extracted Features:")
        print(df_with_features)
        print("-" * 50)
        
        print("\nInfo for the new DataFrame:")
        df_with_features.info()
        print("-" * 50)

        print("\nValue counts for some new columns:")
        for col in ['gender', 'event_type', 'stroke', 'unit']:
            if col in df_with_features:
                print(f"\nValue counts for {col}:")
                print(df_with_features[col].value_counts(dropna=False))

    except Exception as e:
        print(f"An error occurred: {e}")



# %%
# Make sure your DataFrame df_swimming is loaded and available.
# For example:
# df_swimming = pd.read_csv('your_swimming_data.csv') # Or however you load it

# The column in df_swimming that contains the event descriptions
event_column_name = 'event' # Based on your df.info(), this seems to be the correct column

# Check if the column exists to prevent errors
if event_column_name not in df_swimming.columns:
    print(f"Error: Column '{event_column_name}' not found in df_swimming.")
    print(f"Please check the column name. Available columns are: {df_swimming.columns.tolist()}")
else:
    # Apply the feature extraction
    # It's good practice to work on a copy if you want to keep the original df_swimming unchanged
    df_swimming_with_features = extract_all_features(df_swimming.copy(), event_column_name)
    
    # Display the first few rows of the new DataFrame with extracted features
    print("DataFrame with new features (head):")
    print(df_swimming_with_features.head())
    
    # Display info to see the new columns
    print("\nInfo for the DataFrame with new features:")
    df_swimming_with_features.info()
    
    # You can also check value counts for some of the new columns
    print("\nValue counts for 'gender' (new column):")
    print(df_swimming_with_features['gender'].value_counts(dropna=False).head())

    print("\nValue counts for 'event_type' (new column):")
    print(df_swimming_with_features['event_type'].value_counts(dropna=False).head())
    
    print("\nValue counts for 'stroke' (new column):")
    print(df_swimming_with_features['stroke'].value_counts(dropna=False).head())

    # To replace your original df_swimming with the processed one:
    # df_swimming = df_swimming_with_features
    # Or assign it to a new variable name if you prefer.

# %%
df_swimming_with_features.drop(columns=['total_distance_meters', 'relay_leg_distance_meters'], inplace=True)

# %%
df_swimming_with_features.info()


