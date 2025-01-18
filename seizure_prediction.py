import os
import torch
import numpy as np
import h5py
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse

# Import your model and utility functions
from model import DCRNNModel_nextTimePred
import utils

# Define args using your updated get_args() function
def get_args():
    parser = argparse.ArgumentParser('Train DCRNN on TUH data.')

    # General args
    parser.add_argument('--save_dir',
                        type=str,
                        default=None,
                        help='Directory to save the outputs and checkpoints.')
    parser.add_argument(
        '--load_model_path',
        type=str,
        default=None,
        help='Model checkpoint to start training/testing from.')
    parser.add_argument('--do_train',
                        default=False,
                        action='store_true',
                        help='Whether perform training.')
    parser.add_argument('--rand_seed',
                        type=int,
                        default=123,
                        help='Random seed.')
    parser.add_argument(
        '--task',
        type=str,
        default='detection',
        choices=(
            'detection',
            'classification',
            'SS pre-training'),
        help="Seizure detection, seizure type classification, \
                            or SS pre-training.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help='Whether to fine-tune pre-trained model.')

    # Input args
    parser.add_argument(
        '--graph_type',
        choices=(
            'individual',
            'combined'),
        default='individual',
        help='Whether use individual graphs (cross-correlation) or combined graph (distance).')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=60,
                        help='Maximum sequence length in seconds.')
    parser.add_argument(
        '--output_seq_len',
        type=int,
        default=12,
        help='Output seq length for SS pre-training, in seconds.')
    parser.add_argument('--time_step_size',
                        type=int,
                        default=1,
                        help='Time step size in seconds.')
    parser.add_argument('--input_dir',
                        type=str,
                        default=None,
                        help='Dir to resampled EEG signals (.h5 files).')
    parser.add_argument('--raw_data_dir',
                        type=str,
                        default=None,
                        help='Dir to TUH data with raw EEG signals.')
    parser.add_argument('--preproc_dir',
                        type=str,
                        default=None,
                        help='Dir to preprocessed (Fourier transformed) data.')
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Top-k neighbors of each node to keep, for graph sparsity.')

    # Model args
    parser.add_argument("--model_name", type=str, default="dcrnn", choices=("dcrnn", "lstm", "densecnn", "cnnlstm"))
    parser.add_argument('--num_nodes',
                        type=int,
                        default=19,
                        help='Number of nodes in graph.')
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=3,  # Set to 3 to match the pre-trained model
                        help='Number of RNN layers in encoder and/or decoder.')
    parser.add_argument(
        '--pretrained_num_rnn_layers',
        type=int,
        default=3,
        help='Number of RNN layers in encoder and decoder for SS pre-training.')
    parser.add_argument('--rnn_units',
                        type=int,
                        default=64,
                        help='Number of hidden units in DCRNN.')
    parser.add_argument('--dcgru_activation',
                        type=str,
                        choices=('relu', 'tanh'),
                        default='tanh',
                        help='Nonlinear activation used in DCGRU cells.')
    parser.add_argument('--input_dim',
                        type=int,
                        default=100,
                        help='Input seq feature dim.')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=1,
        help='Number of classes for seizure detection/classification.')
    parser.add_argument('--output_dim',
                        type=int,
                        default=100,
                        help='Output seq feature dim.')
    parser.add_argument('--max_diffusion_step',
                        type=int,
                        default=2,
                        help='Maximum diffusion step.')
    parser.add_argument('--cl_decay_steps',
                        type=int,
                        default=3000,
                        help='Scheduled sampling decay steps.')
    parser.add_argument(
        '--use_curriculum_learning',
        default=False,
        action='store_true',
        help='Whether to use curriculum training for seq-seq model.')
    parser.add_argument(
        '--use_fft',
        default=False,
        action='store_true',
        help='Whether the input data is Fourier transformed EEG signal or raw EEG.')

    # Training/test args
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=40,
                        help='Training batch size.')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='Dev/test batch size.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='Dropout rate for dropout layer before final FC.')
    parser.add_argument('--eval_every',
                        type=int,
                        default=1,
                        help='Evaluate on dev set every x epoch.')
    parser.add_argument(
        '--metric_name',
        type=str,
        default='auroc',
        choices=(
            'F1',
            'acc',
            'loss',
            'auroc'),
        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--lr_init',
                        type=float,
                        default=3e-4,
                        help='Initial learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=5e-4,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs for training.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--metric_avg',
                        type=str,
                        default='weighted',
                        help='weighted, micro or macro.')
    parser.add_argument('--data_augment',
                        default=False,
                        action='store_true',
                        help='Whether perform data augmentation.')
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of epochs of patience before early stopping.')

    args = parser.parse_args()

    # which metric to maximize
    if args.metric_name == 'loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ('F1', 'acc', 'auroc'):
        # Best checkpoint is the one that maximizes F1 or acc
        args.maximize_metric = True
    else:
        raise ValueError(
            'Unrecognized metric name: "{}"'.format(
                args.metric_name))

    # must provide load_model_path if testing only
    if (args.load_model_path is None) and not(args.do_train):
        raise ValueError(
            'For evaluation only, please provide trained model checkpoint in argument load_model_path.')

    # filter type
    if args.graph_type == "individual":
        args.filter_type = "dual_random_walk"
    if args.graph_type == "combined":
        args.filter_type = "laplacian"

    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args

# Main function
def main():
    # Get arguments
    args = get_args()

    # Set random seed
    torch.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    # Check and set necessary arguments
    if args.load_model_path is None:
        args.load_model_path = './pretrained_correlation_graph_60s.pth.tar'  # Adjust path as necessary
    if args.input_dir is None:
        args.input_dir = './preprocessed'  # Adjust path as necessary

    # Ensure use_fft is True since input_dim is 100 (FFT-transformed data)
    args.use_fft = True

    # Initialize the model
    model = DCRNNModel_nextTimePred(device=args.device, args=args)

    # Load the pre-trained model checkpoint
    model = utils.load_model_checkpoint(args.load_model_path, model)
    model = model.to(args.device)
    model.eval()

    # Load adjacency matrices (supports)
    adj_mx_path = './adj_mx_3d.pkl'  # Adjust path as necessary
    with open(adj_mx_path, 'rb') as f:
        adj_mx = pickle.load(f)

    # Extract the adjacency matrix from adj_mx
    adjacency_matrix = adj_mx[2]  # Assuming the third element is the adjacency matrix

    # Convert the adjacency matrix to a tensor and move to the device
    supports = [torch.tensor(adjacency_matrix).to(args.device)]

    # Custom Dataset for Inference with FFT preprocessing
    class EEGInferenceDataset(Dataset):
        def __init__(self, data_dir, seq_len, input_dim, standard_channels):
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
            self.data_dir = data_dir
            self.seq_len = seq_len
            self.input_dim = input_dim
            self.standard_channels = [ch.lower() for ch in standard_channels]

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            filename = self.file_list[idx]
            file_path = os.path.join(self.data_dir, filename)
            with h5py.File(file_path, 'r') as f:
                data = f['data'][:]  # Shape: (n_channels, n_samples)
                channels = f['channels'][:]
                channels = [ch.decode('utf-8').lower() for ch in channels]  # Standardize channel names to lower case

            # Create a mapping from channel names to indices
            channel_indices = {ch: idx for idx, ch in enumerate(channels)}

            # Reorder and select channels based on standard_channels
            selected_indices = []
            for ch in self.standard_channels:
                if ch in channel_indices:
                    selected_indices.append(channel_indices[ch])
                else:
                    # Handle missing channels (e.g., pad with zeros)
                    selected_indices.append(None)

            # Extract and reorder data
            reordered_data = []
            for idx in selected_indices:
                if idx is not None:
                    reordered_data.append(data[idx, :])
                else:
                    # Create a zero array if channel is missing
                    reordered_data.append(np.zeros(data.shape[1]))
            data = np.array(reordered_data)  # Shape: (num_nodes, n_samples)

            # Transpose data to (n_samples, num_nodes)
            data = data.T

            # Apply FFT
            data_fft = np.fft.fft(data, n=args.input_dim * 2, axis=0)  # Compute FFT along the time axis
            data_fft = np.abs(data_fft[:args.input_dim, :])  # Take first input_dim frequency bins (magnitude)

            # Ensure data length is at least seq_len
            if data_fft.shape[0] < args.max_seq_len:
                # Pad with zeros if necessary
                pad_length = args.max_seq_len - data_fft.shape[0]
                data_fft = np.pad(data_fft, ((0, pad_length), (0, 0)), 'constant')
            else:
                data_fft = data_fft[:args.max_seq_len, :]  # Truncate to seq_len

            # Convert data to torch tensor
            data_fft = torch.tensor(data_fft, dtype=torch.float32)  # Shape: (seq_len, num_nodes)
            # Add input_dim dimension
            data_fft = data_fft.unsqueeze(-1)  # Shape: (seq_len, num_nodes, input_dim=1)

            return data_fft, filename

    # Standard 19 EEG channels (ensure matching case)
    standard_channels = [
        'fp1', 'fp2', 'f3', 'f4', 'c3', 'c4', 'p3', 'p4', 'o1', 'o2',
        'f7', 'f8', 't3', 't4', 't5', 't6', 'fz', 'cz', 'pz'
    ]

    # Create DataLoader
    data_dir = args.input_dir  # Your preprocessed data directory
    seq_len = args.max_seq_len
    dataset = EEGInferenceDataset(data_dir, seq_len, args.input_dim, standard_channels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run Inference
    predictions = []

    with torch.no_grad():
        for data, filename in tqdm(dataloader, desc="Running Inference"):
            data = data.to(args.device)  # Shape: (batch_size, seq_len, num_nodes, input_dim)

            # Prepare output placeholder
            y_placeholder = torch.zeros((data.size(0), args.output_seq_len, data.size(2), args.output_dim)).to(args.device)

            # Get predictions
            seq_preds = model(data, y_placeholder, supports)

            # Process seq_preds to obtain seizure prediction
            # Example: Compute mean across time, nodes, and output dimensions
            preds = seq_preds.mean(dim=(1, 2, 3))  # Shape: (batch_size,)
            preds = torch.sigmoid(preds)  # Apply sigmoid if necessary
            pred_labels = (preds > 0.5).int()

            # Store predictions
            predictions.append((filename[0], pred_labels.item()))

    # Display predictions
    for filename, pred in predictions:
        label = 'Seizure' if pred == 1 else 'Non-Seizure'
        print(f"File: {filename}, Prediction: {label}")

if __name__ == '__main__':
    main()
