import h5py
import numpy as np

h5_file_path = 'preprocessed/fd036e2f-52a3-409a-b08d-53a877e75e5a.h5'

with h5py.File(h5_file_path, 'r') as h5f:
    data = h5f['data'][:]
    times = h5f['times'][:]
    channels = [ch.decode('utf-8') for ch in h5f['channels'][:]]
    age = h5f['age'][()]
    sex = h5f['sex'][()].decode('utf-8')

print(f"Age: {age}, Sex: {sex}")
print(f"Channels: {channels}")
print(f"Data shape: {data.shape}")
