import mne

# Replace with the path to your FIF file
fif_file_path = "../Abbotsford/0a2fafb5-37fe-4590-a85d-7430b78ba9da_raw.fif"

# Load the raw data from the FIF file
raw = mne.io.read_raw_fif(fif_file_path, preload=True)

# Extract annotations if available
annotations = raw.annotations

# Extract channel information
channels = raw.info['ch_names']

# Extract other info (sampling rate, number of channels, etc.)
sfreq = raw.info['sfreq']
n_channels = len(channels)

# Write annotations to a text file
with open("annotations.txt", "w") as f:
    if annotations is not None:
        f.write(f"Annotations:\n")
        for annot in annotations:
            f.write(f"Description: {annot['description']}, Start: {annot['onset']}, Duration: {annot['duration']}\n")
    else:
        f.write("No annotations found.\n")

# Write channel and other information to a text file
with open("channel_info.txt", "w") as f:
    f.write(f"Number of channels: {n_channels}\n")
    f.write(f"Sampling frequency: {sfreq} Hz\n")
    f.write(f"Channels:\n")
    for ch in channels:
        f.write(f"{ch}\n")

print("Annotations and channel info have been saved to 'annotations.txt' and 'channel_info.txt'.")
