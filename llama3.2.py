import os
import h5py
import numpy as np
from scipy.signal import welch
import json
from transformers import pipeline
from tqdm import tqdm  # For progress bar

# Define directories and constants
PREPROCESSED_DIR = "preprocessed"
OUTPUT_DIR = "predictions"
MAX_FILES = 1  # Process only the first 5 files

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define label mapping
LABEL_MAPPING = {
    "Seizure": "Seizure",
    "No Seizure": "No Seizure"
}

# Function to extract features from EEG data
def extract_features(data, fs=250):
    """
    Extracts both time-domain and frequency-domain features.
    """
    features = {}
    # Time-domain features
    features['mean'] = np.mean(data, axis=1).tolist()
    features['variance'] = np.var(data, axis=1).tolist()
    features['skewness'] = np.mean(((data - np.mean(data, axis=1, keepdims=True))**3), axis=1).tolist()
    features['kurtosis'] = np.mean(((data - np.mean(data, axis=1, keepdims=True))**4), axis=1).tolist()

    # Frequency-domain features using Welch's method
    freqs, psd = welch(data, fs=fs, nperseg=fs*2)
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        features[band] = np.mean(psd[:, idx], axis=1).tolist()
    return features

# Function to preprocess and format data from H5 files
def preprocess_h5_files(folder, max_files):
    """
    Loads and preprocesses the first N H5 files from the given folder.
    """
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")]
    data_list = []
    for i, file_path in enumerate(tqdm(files[:max_files], desc="Preprocessing Files")):
        with h5py.File(file_path, 'r') as h5f:
            data = h5f['data'][:]  # EEG signal (channels x time)
            channels = [ch.decode() for ch in h5f['channels'][:]]  # Channel names
            age = h5f['age'][()]
            sex = h5f['sex'][()].decode()
            
            # Extract features
            features = extract_features(data)

            # Format into structured JSON
            formatted_data = {
                "file": os.path.basename(file_path),
                "age": age,
                "sex": sex,
                "channels": channels,
                "features": features
            }
            data_list.append(formatted_data)
    return data_list

# Function to predict seizures using the Hugging Face LLM
def predict_seizures(data_list):
    """
    Uses a Hugging Face LLM to predict whether there is a seizure in the EEG data and explain the decision.
    """
    model = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device=-1)  # Use CPU

    predictions = []

    for data in tqdm(data_list, desc="Predicting Seizures"):
        # Simplified prompt
        prompt = f"""
        Patient Age: {data['age']}
        Patient Sex: {data['sex']}
        EEG Features Summary:
        {json.dumps(data['features'], indent=2)}

        Based on the features, predict if the patient is experiencing a seizure. Respond with "Seizure" or "No Seizure" and explain your reasoning.
        Format your response as:
        Prediction: [Seizure/No Seizure]
        Explanation: [Reason for the prediction based on features]
        """

        # Generate prediction and explanation
        result = model(prompt, max_new_tokens=300, truncation=True, do_sample=True)

        output = result[0]['generated_text']

        # Parse the prediction and explanation
        prediction = "Unknown"
        explanation = "No explanation provided."

        if "Prediction:" in output:
            prediction = output.split("Prediction:")[1].split("Explanation:")[0].strip()
        if "Explanation:" in output:
            explanation = output.split("Explanation:")[1].strip()

        predictions.append({
            "file": data['file'],
            "prediction": prediction,  # Extracted prediction
            "explanation": explanation  # Extracted explanation
        })
    return predictions


# Main execution
if __name__ == "__main__":
    # Preprocess the files
    print("Preprocessing H5 files...")
    preprocessed_data = preprocess_h5_files(PREPROCESSED_DIR, MAX_FILES)

    # Predict seizures
    print("Predicting seizures using the LLM...")
    predictions = predict_seizures(preprocessed_data)

    # Save predictions to JSON file
    output_file = os.path.join(OUTPUT_DIR, "predictions_with_explanations.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)

    # Save individual explanations
    for prediction in predictions:
        individual_file = os.path.join(OUTPUT_DIR, f"{prediction['file']}_explanation.json")
        with open(individual_file, "w") as f:
            json.dump(prediction, f, indent=4)

    print(f"Predictions and explanations saved to {OUTPUT_DIR}")

    # Display results
    for result in predictions:
        print(f"File: {result['file']} - Prediction: {result['prediction']}")
        print(f"Explanation: {result['explanation']}\n")
