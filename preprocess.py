import os
import mne
import pandas as pd
import numpy as np
import h5py
import re

# Paths
fif_folder = "your_eeg_data_folder"
csv_file = "your_eeg_label_csv_file"
output_folder = "preprocessed"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Ensure 'ScanID', 'Age', and 'Sex' columns are present
required_columns = ['ScanID', 'Age', 'Sex']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV file must contain the columns: {required_columns}")

# Remove entries with missing 'Age' or 'Sex'
df = df.dropna(subset=['Age', 'Sex'])

# Convert 'ScanID' to string for consistency
df['ScanID'] = df['ScanID'].astype(str)

# Function to parse 'Age' strings
def parse_age(age_str):
    """
    Parses an age string and returns the age in years as a float rounded to two decimal places.
    Examples:
        '75' -> 75.0
        '3Y 06M' -> 3.5
        '6M' -> 0.5
    """
    if pd.isna(age_str):
        return np.nan

    age_str = str(age_str).strip()
    if age_str.isdigit():
        # Age is a simple integer
        return float(age_str)
    else:
        # Parse age string with years and months
        years = 0
        months = 0
        # Match patterns like '3Y' or '06M'
        year_match = re.search(r'(\d+)\s*[Yy]', age_str)
        month_match = re.search(r'(\d+)\s*[Mm]', age_str)

        if year_match:
            years = int(year_match.group(1))
        if month_match:
            months = int(month_match.group(1))

        age_in_years = years + months / 12.0
        # Round to two decimal places
        age_in_years = round(age_in_years, 2)
        return age_in_years if age_in_years > 0 else np.nan

# Apply the parse_age function to the 'Age' column
df['AgeYears'] = df['Age'].apply(parse_age)

# Remove entries where 'AgeYears' is missing or NaN
df = df.dropna(subset=['AgeYears'])

# Create a mapping from ScanID to AgeYears and Sex
scanid_info = df.set_index('ScanID')[['AgeYears', 'Sex']].to_dict('index')

# Iterate over FIF files in the folder
for file_name in os.listdir(fif_folder):
    if file_name.endswith(".fif"):
        fif_file_path = os.path.join(fif_folder, file_name)
        scan_id = file_name.replace("_raw.fif", "")

        # Check if ScanID exists in the CSV mapping
        if scan_id in scanid_info:
            age = scanid_info[scan_id]['AgeYears']
            sex = scanid_info[scan_id]['Sex']

            # Check if Age and Sex are not missing
            if pd.isna(age) or pd.isna(sex):
                print(f"Skipping {file_name}: Missing Age or Sex.")
                continue

            try:
                # Load the FIF file
                print(f"Processing {file_name}...")
                raw = mne.io.read_raw_fif(fif_file_path, preload=True, verbose=False)

                # Select only EEG channels
                raw.pick_types(eeg=True, exclude=[])

                # Resample to 250 Hz
                raw.resample(250, npad="auto")

                # Extract the first 10 seconds of data
                start_time = 0  # in seconds
                end_time = 10   # in seconds
                raw.crop(tmin=start_time, tmax=end_time)

                # Get the data and channel names
                data, times = raw.get_data(return_times=True)
                channel_names = raw.info['ch_names']

                # Prepare data for saving
                h5_file_path = os.path.join(output_folder, f"{scan_id}.h5")
                with h5py.File(h5_file_path, 'w') as h5f:
                    h5f.create_dataset('data', data=data)
                    h5f.create_dataset('times', data=times)
                    h5f.create_dataset('channels', data=np.array(channel_names, dtype='S'))
                    h5f.create_dataset('age', data=age)
                    h5f.create_dataset('sex', data=np.string_(sex))

                print(f"Saved preprocessed data to {h5_file_path}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"Skipping {file_name}: ScanID not found in CSV mapping.")
