import h5py
import numpy as np
import os

os.makedirs('data/H5', exist_ok=True)
with h5py.File('data/H5/test_das.h5', 'w') as f:
    # Create 30 seconds of data at 1kHz, 50 channels
    # 30 * 1000 = 30000 points
    data = np.random.randn(30000, 50).astype(np.float32) * 0.1

    # Add some "anomalies"
    # Anomaly 1: channel 10-15, time 5000-5200
    data[5000:5200, 10:15] += 5.0

    # Anomaly 2: channel 30-35, time 15000-15300
    data[15000:15300, 30:35] += 10.0

    # st-yolo-das usually reads blocks or keys. In styolo2.py it expects keys.
    # blocks = [hf[k][:].copy() for k in keys]
    f.create_dataset('0', data=data)

print("Created dummy DAS H5 file.")
