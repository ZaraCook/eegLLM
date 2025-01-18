import os
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import Captum for Integrated Gradients
from captum.attr import IntegratedGradients
from datetime import datetime  # For including the time in the report

# Import scipy.signal for spectral analysis
from scipy.signal import welch

# Set random seed for reproducibility
random_seed = 1234
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Parameters
data_folder = 'preprocessed'  # Folder containing H5 files
batch_size = 64
num_epochs = 50
learning_rate = 0.001
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
patience = 10  # Early stopping patience
weight_decay = 1e-4  # L2 regularization
sequence_length = 50  # Length of the sequences for LSTM
num_reports = 10  # Number of reports to generate from the test set

# Sampling frequency (Hz) of the EEG data
fs = 256  # Replace with your actual sampling rate

# List of EEG channel names in the exact order as in the data
channel_names = [
    'C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2',
    'Fpz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6'
]

# Map 'F' and 'M' to 0 and 1
sex_mapping = {'F': 0, 'M': 1}
gender_mapping = {0: 'Female', 1: 'Male'}

# Custom Dataset Class for Gender Classification
class EEGGenderDataset(Dataset):
    def __init__(self, data_folder, sequence_length):
        self.data_folder = data_folder
        self.sequence_length = sequence_length
        self.sex_mapping = sex_mapping
        # Exclude hidden files and ensure only HDF5 files are included
        self.file_list = [
            f for f in os.listdir(data_folder)
            if f.endswith('.h5') and not f.startswith('.')
        ]
        self.data = []
        self.labels = []
        self.filenames = []
        self.load_data()

    def load_data(self):
        print("Loading data and generating sequences...")
        sequence_length = self.sequence_length
        for file_name in tqdm(self.file_list, desc="Files Loaded"):
            file_path = os.path.join(self.data_folder, file_name)
            try:
                with h5py.File(file_path, 'r') as h5f:
                    if 'sex' in h5f and 'data' in h5f:
                        data = h5f['data'][:]  # Shape: (n_channels, n_times)
                        sex = h5f['sex'][()].decode('utf-8')
                        label = self.sex_mapping.get(sex)
                        if label is not None:
                            # Only select the first 20 channels (EEG channels)
                            data = data[:20, :]
                            # Normalize per channel
                            data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
                            num_steps = data.shape[1]  # Number of time steps
                            # Generate sequences
                            for start in range(0, num_steps - sequence_length + 1, sequence_length):
                                end = start + sequence_length
                                seq = data[:, start:end]  # Shape: (n_channels, sequence_length)
                                # Transpose to (sequence_length, n_channels)
                                seq = seq.T  # Shape: (sequence_length, n_channels)
                                self.data.append(seq)
                                self.labels.append(label)
                                self.filenames.append(file_name)
                    else:
                        print(f"File {file_path} does not contain 'sex' or 'data' dataset. Skipping.")
            except OSError as e:
                print(f"Error opening file {file_path}: {e}. Skipping.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]  # Shape: (sequence_length, n_channels)
        label = self.labels[idx]
        filename = self.filenames[idx]
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return sample, label, filename

# Initialize the dataset
dataset = EEGGenderDataset(data_folder, sequence_length)

# Check if dataset is empty
if len(dataset) == 0:
    raise ValueError("No valid data samples were loaded. Please check your data files.")

# Determine input dimensions
n_samples = len(dataset)
n_channels = dataset[0][0].shape[1]  # Number of channels

# Split into training, validation, and testing sets
train_size = int(train_ratio * n_samples)
val_size = int(val_ratio * n_samples)
test_size = n_samples - train_size - val_size

# Ensure the sum of splits equals the total number of samples
assert train_size + val_size + test_size == n_samples, "Split sizes do not sum up to total samples."

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)  # Batch size 1 for per-sample processing

# Define the LSTM model
class GenderLSTM(nn.Module):
    def __init__(self, input_size, sequence_length, num_classes=1):
        super(GenderLSTM, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length

        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(sequence_length)

        self.lstm2 = nn.LSTM(512, 128, batch_first=True)
        self.bn2 = nn.BatchNorm1d(sequence_length)

        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.bn3 = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm1(x)  # out shape: (batch_size, sequence_length, 512)
        out = self.dropout1(out)
        out = self.bn1(out)
        out, _ = self.lstm2(out)  # out shape: (batch_size, sequence_length, 128)
        out = self.bn2(out)
        out, _ = self.lstm3(out)  # out shape: (batch_size, sequence_length, 64)
        out = out[:, -1, :]  # Take the last time step
        out = self.bn3(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Initialize the model
input_size = n_channels  # Number of features per time step
model = GenderLSTM(input_size, sequence_length)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer with weight decay (L2 regularization)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Early stopping variables
best_val_loss = np.inf
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

# Training loop with validation and early stopping
print("Starting training...")
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
    for inputs, labels, _ in train_loader_tqdm:
        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1, 1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Update progress bar
        train_loader_tqdm.set_postfix({'Loss': loss.item()})

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            preds = (outputs > 0.5).float()
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)

    # Scheduler step
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Training Loss: {epoch_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}, "
          f"Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

print("Training completed.")

# Load best model weights
model.load_state_dict(best_model_wts)
model.eval()  # Set model to evaluation mode

# Test set evaluation
print("Evaluating on test set...")
test_loss = 0.0
all_test_preds = []
all_test_labels = []
with torch.no_grad():
    for inputs, labels, _ in tqdm(test_loader, desc="Testing"):
        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1, 1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

        preds = (outputs > 0.5).float()
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_loader.dataset)
test_accuracy = accuracy_score(all_test_labels, all_test_preds)
precision = precision_score(all_test_labels, all_test_preds)
recall = recall_score(all_test_labels, all_test_preds)
f1 = f1_score(all_test_labels, all_test_preds)

print(f"Test Loss: {test_loss:.4f}, "
      f"Test Accuracy: {test_accuracy:.4f}, "
      f"Precision: {precision:.4f}, "
      f"Recall: {recall:.4f}, "
      f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_test_labels, all_test_preds)
print("Confusion Matrix:")
print(cm)

# Initialize Integrated Gradients for Explainability
ig = IntegratedGradients(model)

# Initialize Hugging Face Token and models
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
if HUGGINGFACE_TOKEN is None:
    raise ValueError("Please set your Hugging Face token as an environment variable 'HUGGINGFACE_TOKEN'.")

# Configure the model to use the token
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Update to a model you have access to

print("Loading Llama model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)

# Ensure that tokenizer has pad_token_id set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with reduced precision and device mapping
llama_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=HUGGINGFACE_TOKEN,
    torch_dtype=torch.float16,
    device_map='auto'
)
llama_model.eval()

# Create reports directory
reports_dir = 'reports_gender'
os.makedirs(reports_dir, exist_ok=True)

# Randomly select samples from the test set for report generation
test_indices = list(range(len(test_dataset)))
random.shuffle(test_indices)
selected_indices = test_indices[:num_reports]

# Subset the test dataset for report generation
report_dataset = torch.utils.data.Subset(test_dataset, selected_indices)
report_loader = DataLoader(report_dataset, batch_size=1, shuffle=False, drop_last=False)

# Testing and report generation
print("Generating reports for selected test samples...")
report_loader_tqdm = tqdm(report_loader, desc="Generating Reports")

with torch.no_grad():
    for inputs, labels, filenames in report_loader_tqdm:
        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1, 1)
        filename = filenames[0]  # Since batch size is 1

        # Get the prediction
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        pred_gender = gender_mapping[int(preds.item())]
        true_gender = gender_mapping[int(labels.item())]

        # Explainability: Generate explanations using Integrated Gradients
        inputs.requires_grad = True

        # Compute attributions
        attributions, delta = ig.attribute(
            inputs,
            target=None,  # For binary classification, target can be None
            return_convergence_delta=True
        )

        # Summarize the attributions
        attributions_sum = attributions.sum(dim=1).squeeze().cpu().numpy()  # Sum over time
        attributions_mean = attributions_sum  # Since we already summed over time

        # Identify top contributing channels
        top_indices = np.argsort(-np.abs(attributions_mean))[:5]  # Top 5 channels by absolute attribution
        top_channels = top_indices.tolist()
        # Map channel indices to names
        top_channel_names = [channel_names[i] for i in top_channels]

        # Get summary statistics and spectral features for top channels
        eeg_file_path = os.path.join(data_folder, filename)
        with h5py.File(eeg_file_path, 'r') as h5f:
            eeg_data = h5f['data'][:20, :]  # Only the first 20 channels
            # Normalize per channel
            eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)
            eeg_summary = ""
            for i in top_channels:
                channel_data = eeg_data[i]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                max_val = np.max(channel_data)
                min_val = np.min(channel_data)

                # Compute power spectral density
                freqs, psd = welch(channel_data, fs=fs, nperseg=fs*2)  # Using segments of 2 seconds

                # Define frequency bands
                delta_band = (0.5, 4)
                theta_band = (4, 8)
                alpha_band = (8, 13)
                beta_band = (13, 30)
                gamma_band = (30, 45)

                # Compute band powers
                delta_power = np.trapz(psd[(freqs >= delta_band[0]) & (freqs < delta_band[1])],
                                       freqs[(freqs >= delta_band[0]) & (freqs < delta_band[1])])
                theta_power = np.trapz(psd[(freqs >= theta_band[0]) & (freqs < theta_band[1])],
                                       freqs[(freqs >= theta_band[0]) & (freqs < theta_band[1])])
                alpha_power = np.trapz(psd[(freqs >= alpha_band[0]) & (freqs < alpha_band[1])],
                                       freqs[(freqs >= alpha_band[0]) & (freqs < alpha_band[1])])
                beta_power = np.trapz(psd[(freqs >= beta_band[0]) & (freqs < beta_band[1])],
                                      freqs[(freqs >= beta_band[0]) & (freqs < beta_band[1])])
                gamma_power = np.trapz(psd[(freqs >= gamma_band[0]) & (freqs < gamma_band[1])],
                                       freqs[(freqs >= gamma_band[0]) & (freqs < gamma_band[1])])

                # Normalize band powers by total power
                total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
                delta_rel_power = delta_power / total_power
                theta_rel_power = theta_power / total_power
                alpha_rel_power = alpha_power / total_power
                beta_rel_power = beta_power / total_power
                gamma_rel_power = gamma_power / total_power

                eeg_summary += (f"Channel {channel_names[i]} - Mean: {mean:.2f}, Std: {std:.2f}, "
                                f"Max: {max_val:.2f}, Min: {min_val:.2f}\n"
                                f"Relative Band Powers: Delta: {delta_rel_power:.2f}, Theta: {theta_rel_power:.2f}, "
                                f"Alpha: {alpha_rel_power:.2f}, Beta: {beta_rel_power:.2f}, Gamma: {gamma_rel_power:.2f}\n\n")

        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the report header
        report_header = f"""Scan Name: "{filename}"
Time this report was written: "{current_time}"
Actual Gender: {true_gender}
Predicted Gender: {pred_gender}
Most Important Channels: {top_channel_names}

EEG Data Summary for Top Channels:
{eeg_summary}
"""

        # Prepare the prompt for the Llama model
        prompt = f"""
You are a medical professional analyzing EEG data.

Based on the following information, write a medical report explaining how the EEG findings relate to the patient's gender.

{report_header}

Provide an interpretation of the EEG activity in the top channels, including the relative band powers (Delta, Theta, Alpha, Beta, Gamma), and discuss how it correlates with the patient's gender. Explain any possible reasons for discrepancies between the predicted and actual gender if applicable. Focus on the EEG patterns typically associated with the patient's gender. Keep the report professional and avoid mentioning any machine learning models or algorithms.

Ensure that the report ends with a complete sentence and is concise.
        """

        # Tokenize and generate the report
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(llama_model.device)

        # Generate text with the Llama model
        output_ids = llama_model.generate(
            input_ids=input_ids,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.0,
            num_beams=1
        )

        # Decode the generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Since the input prompt is included in the output, remove it
        generated_response = generated_text[len(prompt):].strip()

        # Ensure the report ends with a complete sentence
        if not generated_response.endswith('.'):
            generated_response += '.'

        # Combine the report header and LLM output
        final_report = report_header + "\n" + generated_response

        # Save the report to a file
        report_filename = f"{os.path.splitext(filename)[0]}_report.txt"
        report_path = os.path.join(reports_dir, report_filename)
        with open(report_path, 'w') as f:
            f.write(final_report)

print("Reports generated in the 'reports_gender' directory.")
