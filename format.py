import os
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

# Path to the folder containing the .h5 files
h5_folder = "preprocessed"  # Replace with your folder name
output_csv = "data.csv"  # Name of the output CSV file

# List to store data for the DataFrame
data_rows = []

# Define the label for this dataset (use 0 for non-epileptic or 1 for epileptic seizure based on context)
default_label = 0  # Update this if needed based on your dataset

# Get a list of all .h5 files in the folder
h5_files = [f for f in os.listdir(h5_folder) if f.endswith(".h5")]

# Iterate over all .h5 files with a progress bar
for file_name in tqdm(h5_files, desc="Processing .h5 files"):
    h5_file_path = os.path.join(h5_folder, file_name)
    try:
        with h5py.File(h5_file_path, 'r') as h5f:
            # Extract EEG data
            data = h5f['data'][:]
            
            # Log the shape of the data
            #print(f"Processing {file_name}: Data shape {data.shape}")
            
            # Check if the shape is compatible
            num_channels, num_points = data.shape
            if num_points < 178:
                print(f"Skipping {file_name}: Not enough time points for a single chunk.")
                continue

            # Calculate the number of full chunks
            chunk_size = 178
            num_chunks = num_points // chunk_size

            for i in range(num_chunks):
                chunk = data[:, i * chunk_size:(i + 1) * chunk_size]
                # Flatten the chunk to a 1D array
                flattened_chunk = chunk.flatten()
                # Append the flattened chunk with the default label
                data_rows.append(np.append(flattened_chunk, default_label))

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Check if data_rows has valid entries
if not data_rows:
    raise ValueError("No valid data rows extracted. Check input files and chunking logic.")

# Convert the list of data rows into a DataFrame
columns = [f"X{i+1}" for i in range(178)] + ["y"]
df = pd.DataFrame(data_rows, columns=columns)

# Save to CSV
df.to_csv(output_csv, index=False)

print(f"CSV file created: {output_csv}")
