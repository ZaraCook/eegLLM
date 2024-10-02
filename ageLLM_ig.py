import os
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from braindecode.models import EEGConformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import Captum for Integrated Gradients
from captum.attr import IntegratedGradients
from datetime import datetime  # For including the time in the report

# Parameters
data_folder = 'preprocessed'  # Folder containing H5 files
batch_size = 64
num_epochs = 100
learning_rate = 0.001
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
random_seed = 42
patience = 10  # Early stopping patience
weight_decay = 1e-4  # L2 regularization
num_reports = 10  # Number of reports to generate from the test set

# Set random seed for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# List of EEG channel names in the exact order as in the data
channel_names = [
    'C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2',
    'Fpz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6'
]

# Custom Dataset Class for Age Regression
class EEGAgeDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
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
        print("Loading data...")
        for file_name in tqdm(self.file_list, desc="Files Loaded"):
            file_path = os.path.join(self.data_folder, file_name)
            try:
                with h5py.File(file_path, 'r') as h5f:
                    # Check if 'age' dataset exists
                    if 'age' in h5f and 'data' in h5f:
                        data = h5f['data'][:]  # Shape: (n_channels, n_times)
                        age = h5f['age'][()]
                        if age is not None:
                            # Only select the first 20 channels (EEG channels)
                            data = data[:20, :]
                            # Ensure the channel order matches the channel_names list
                            if data.shape[0] != len(channel_names):
                                print(f"Channel mismatch in file {file_name}. Skipping.")
                                continue
                            self.data.append(data)
                            self.labels.append(age)
                            self.filenames.append(file_name)
                    else:
                        print(f"File {file_path} does not contain 'age' or 'data' dataset. Skipping.")
            except OSError as e:
                print(f"Error opening file {file_path}: {e}. Skipping.")
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(f"Total samples loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]  # Shape: (n_channels, n_times)
        label = self.labels[idx]
        filename = self.filenames[idx]
        # Normalize the sample per channel
        sample = (sample - np.mean(sample, axis=1, keepdims=True)) / np.std(sample, axis=1, keepdims=True)
        # Convert to torch tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return sample, label, filename

# Initialize the dataset
dataset = EEGAgeDataset(data_folder)

# Check if dataset is empty
if len(dataset) == 0:
    raise ValueError("No valid data samples were loaded. Please check your data files.")

# Determine input dimensions
n_samples = len(dataset)
n_channels, n_times = dataset.data[0].shape

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

# Define the model with adjusted hyperparameters for regression
model = EEGConformer(
    n_outputs=1,            # Single output for regression
    n_chans=n_channels,
    n_times=n_times,
    n_filters_time=32,      # Adjusted to prevent overfitting
    filter_time_length=25,
    pool_time_length=75,
    pool_time_stride=15,
    att_depth=4,            # Adjusted depth
    att_heads=8,            # Adjusted number of heads
    att_drop_prob=0.6,      # Increased dropout rate
    final_fc_length='auto',
    add_log_softmax=False
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer with weight decay (L2 regularization)
criterion = nn.MSELoss()
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
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs).squeeze(-1)  # Shape: [batch_size]
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
            labels = labels.to(device)

            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            all_val_preds.extend(outputs.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_mae = mean_absolute_error(all_val_labels, all_val_preds)
    val_mse = mean_squared_error(all_val_labels, all_val_preds)
    val_rmse = np.sqrt(val_mse)
    val_std_ae = np.std(np.abs(np.array(all_val_labels) - np.array(all_val_preds)))
    val_r2 = r2_score(all_val_labels, all_val_preds)

    # Scheduler step
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Training Loss: {epoch_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}, "
          f"MAE: {val_mae:.2f}, "
          f"MSE: {val_mse:.2f}, "
          f"RMSE: {val_rmse:.2f}, "
          f"STD AE: {val_std_ae:.2f}, "
          f"R2: {val_r2:.2f}")

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
        labels = labels.to(device)

        outputs = model(inputs).squeeze(-1)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

        all_test_preds.extend(outputs.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_loader.dataset)
test_mae = mean_absolute_error(all_test_labels, all_test_preds)
test_mse = mean_squared_error(all_test_labels, all_test_preds)
test_rmse = np.sqrt(test_mse)
test_std_ae = np.std(np.abs(np.array(all_test_labels) - np.array(all_test_preds)))
test_r2 = r2_score(all_test_labels, all_test_preds)

print(f"Test Loss: {test_loss:.4f}, "
      f"MAE: {test_mae:.2f}, "
      f"MSE: {test_mse:.2f}, "
      f"RMSE: {test_rmse:.2f}, "
      f"STD AE: {test_std_ae:.2f}, "
      f"R2: {test_r2:.2f}")

# Initialize Integrated Gradients for Explainability
ig = IntegratedGradients(model)

# Initialize Hugging Face Token and models
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
if HUGGINGFACE_TOKEN is None:
    raise ValueError("Please set your Hugging Face token as an environment variable 'HUGGINGFACE_TOKEN'.")

# Configure the model to use the token
model_name = "meta-llama/Llama-3.2-3B-Instruct"

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
reports_dir = 'reports_test'
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
        labels = labels.to(device)
        filename = filenames[0]  # Since batch size is 1

        # Get the prediction
        outputs = model(inputs).squeeze(-1)
        pred_age = outputs.item()
        true_age = labels.item()

        # Explainability: Generate explanations using Integrated Gradients
        inputs.requires_grad = True

        # Compute attributions
        attributions, delta = ig.attribute(
            inputs,
            target=None,  # For regression tasks
            return_convergence_delta=True
        )

        # Summarize the attributions
        attributions_sum = attributions.sum(dim=2).squeeze().cpu().numpy()  # Sum over time
        attributions_mean = attributions_sum  # Since we already summed over time

        # Identify top contributing channels
        top_indices = np.argsort(-attributions_mean)[:5]  # Top 5 channels
        top_channels = top_indices.tolist()
        # Map channel indices to names
        top_channel_names = [channel_names[i] for i in top_channels]
        explanation = f"The top contributing EEG channels are {top_channel_names}."

        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the prompt for the Llama model
        prompt = f"""
You are a medical AI assistant specialized in EEG analysis.

Based on the following data, generate a concise medical report explaining how the model predicted the patient's age.

- Report Name: {filename}_report.txt
- Time of Report: {current_time}
- EEG File: '{filename}'
- Ground Truth Age: {true_age}
- Predicted Age: {pred_age:.2f}
- Top Contributing EEG Channels: {top_channel_names}

Explain how the top contributing EEG channels influenced the age prediction. Focus on the significance of these channels in EEG analysis related to age. Keep the report concise and accurate. Do not include any false information.

Ensure that the report ends with a complete sentence.
        """

        # Tokenize and generate the report
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(llama_model.device)

        # Generate text with the Llama model
        output_ids = llama_model.generate(
            input_ids=input_ids,
            max_new_tokens=300,
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

        # Save the report to a file
        report_filename = f"{os.path.splitext(filename)[0]}_report.txt"
        report_path = os.path.join(reports_dir, report_filename)
        with open(report_path, 'w') as f:
            f.write(generated_response)

print("Reports generated in the 'reports_test' directory.")
