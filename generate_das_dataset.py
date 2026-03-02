"""
DAS Waterfall to ST-YOLO Format Converter
Improved version of styolo2.py
"""

import os
# =================== Essential Environment Variables ===================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# =====================================================================

import sys
import h5py
import hdf5plugin
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import argparse
from tqdm import tqdm
import glob
import weakref
import cv2
import json
import torch
from dataclasses import dataclass, field
from enum import Enum, auto

# Add st-yolo to sys.path
STYOLO_PATH = str(Path.cwd() / "applications/dvsdetection/st-yolo")
if STYOLO_PATH not in sys.path:
    sys.path.insert(0, STYOLO_PATH)

from data.utils.representations import StackedHistogram
from utils.preprocessing import _blosc_opts

# ======== DAS Data Parameters ========
ORIGINAL_SAMPLING_RATE_HZ = 1000
DOWNSAMPLED_SAMPLING_RATE_HZ = 50
DOWNSAMPLE_FACTOR = ORIGINAL_SAMPLING_RATE_HZ // DOWNSAMPLED_SAMPLING_RATE_HZ # 20

WINDOW_SIZE_SEC = 5.0
WINDOW_SIZE_POINTS_DS = int(WINDOW_SIZE_SEC * DOWNSAMPLED_SAMPLING_RATE_HZ) # 250 points

# For DAS, we typically step by 1 second to create sequences
STEP_SIZE_SEC = 1.0
STEP_SIZE_POINTS_DS = int(STEP_SIZE_SEC * DOWNSAMPLED_SAMPLING_RATE_HZ) # 50 points

NUM_CHANNELS = 50

# Event Representation Parameters
TS_STEP_EV_REPR_MS = 50 # 50ms per bin
EV_REPR_NBINS = 10 # StackedHistogram bins

class H5Writer:
    def __init__(self, outfile: Path, key: str, ev_repr_shape: Tuple, numpy_dtype: np.dtype):
        assert len(ev_repr_shape) == 3
        self.h5f = h5py.File(str(outfile), 'w')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        self.key = key
        self.numpy_dtype = numpy_dtype

        maxshape = (None,) + ev_repr_shape
        chunkshape = (1,) + ev_repr_shape
        self.maxshape = maxshape
        self.h5f.create_dataset(key, dtype=self.numpy_dtype, shape=chunkshape, chunks=chunkshape,
                                maxshape=maxshape, **_blosc_opts(complevel=1, shuffle='byte'))
        self.t_idx = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalizer()

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def close(self):
        self.h5f.close()

    def add_data(self, data: np.ndarray):
        assert data.dtype == self.numpy_dtype
        assert data.shape == self.maxshape[1:]
        new_size = self.t_idx + 1
        self.h5f[self.key].resize(new_size, axis=0)
        self.h5f[self.key][self.t_idx:new_size] = data
        self.t_idx = new_size

class DASSequenceProcessor:
    def __init__(self, output_dir: Path, threshold_percentile: float = 95.0, min_area: int = 4):
        self.output_dir = output_dir
        self.threshold_percentile = threshold_percentile
        self.min_area = min_area
        self.sequence_count = 0

    def downsample(self, data: np.ndarray) -> np.ndarray:
        """Time axis downsampling using Max pooling."""
        H, W = data.shape
        new_H = H // DOWNSAMPLE_FACTOR
        trimmed_H = new_H * DOWNSAMPLE_FACTOR
        data = data[:trimmed_H]
        blocks = data.reshape(new_H, DOWNSAMPLE_FACTOR, W)
        return np.max(blocks, axis=1).astype(np.float32)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Per-channel z-score normalization."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True) + 1e-6
        return (data - mean) / std

    def process_file(self, h5_path: Path, split: str):
        print(f"Processing {h5_path.name}...")
        with h5py.File(h5_path, 'r') as f:
            # Assuming data is in keys that are numeric or a specific 'data' key
            keys = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            if not keys: return

            raw_data = []
            for k in keys:
                raw_data.append(f[k][:].copy())

            full_data = np.abs(np.concatenate(raw_data, axis=0).astype(np.float32))

            # Downsample
            ds_data = self.downsample(full_data)
            # Normalize
            norm_data = self.normalize(ds_data)

            total_points = len(norm_data)

            for start_idx in range(0, total_points - WINDOW_SIZE_POINTS_DS + 1, STEP_SIZE_POINTS_DS):
                window_data = norm_data[start_idx : start_idx + WINDOW_SIZE_POINTS_DS]
                self.create_sequence(window_data, split)

    def create_sequence(self, window_data: np.ndarray, split: str):
        seq_id = self.sequence_count
        self.sequence_count += 1

        seq_dir = self.output_dir / split / f"sequence_{seq_id:06d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        # 1. Detection & Labeling
        # We detect anomalies in the WHOLE window but only label based on the LAST frame's context
        # In ST-YOLO, we typically have one label for one 'representation'
        # Here we follow the logic: The labeled frame is at the END of the window.

        threshold = np.percentile(window_data, self.threshold_percentile)
        binary_mask = window_data > threshold

        # Convert to events (x, y, p, t)
        # y is time in DS points, x is channel
        y_idx, x_idx = np.where(binary_mask)
        # Timestamps in microseconds. 50Hz -> 20ms per point.
        # Let's start sequence at t=0
        t_us = y_idx * (1000000 // DOWNSAMPLED_SAMPLING_RATE_HZ)
        events = {
            'x': x_idx.astype(np.int64),
            'y': y_idx.astype(np.int64),
            'p': np.ones_like(x_idx, dtype=np.int64),
            't': t_us.astype(np.int64)
        }

        # Generate labels for the LAST frame (t = WINDOW_SIZE_SEC)
        # Actually ST-YOLO expects labels at specific timestamps.
        # We'll put a label frame at the very end of the 5s window.
        last_frame_ts_us = int(WINDOW_SIZE_SEC * 1000000)

        # To find labels, we look at the last portion of the mask
        # Let's say any anomaly in the last 200ms counts as a label for the end frame
        lookback_points = int(0.2 * DOWNSAMPLED_SAMPLING_RATE_HZ)
        last_mask_slice = binary_mask[-max(1, lookback_points):]

        labels = []
        if np.any(last_mask_slice):
            # Connected components on the whole window mask to get consistent bboxes
            num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.min_area: continue

                # Check if this component intersects with our "last frame" lookback
                y_start = stats[i, cv2.CC_STAT_TOP]
                y_end = y_start + stats[i, cv2.CC_STAT_HEIGHT]

                if y_end >= (WINDOW_SIZE_POINTS_DS - lookback_points):
                    # This component is active near the end
                    x = float(stats[i, cv2.CC_STAT_LEFT])
                    y = float(stats[i, cv2.CC_STAT_TOP])
                    w = float(stats[i, cv2.CC_STAT_WIDTH])
                    h = float(stats[i, cv2.CC_STAT_HEIGHT])
                    labels.append((last_frame_ts_us, x, y, w, h, 0, 1.0))

        # Save Labels
        labels_dir = seq_dir / "labels_v2"
        labels_dir.mkdir(exist_ok=True)

        dtype = [('t', 'i8'), ('x', 'f4'), ('y', 'f4'), ('w', 'f4'), ('h', 'f4'), ('class_id', 'i4'), ('class_confidence', 'f4')]
        labels_arr = np.array(labels, dtype=dtype)

        # In ST-YOLO, objframe_idx_2_label_idx points to the start index in the 'labels' array for each frame.
        # Since we have only one labeled frame per sequence, it always starts at index 0.
        np.savez(labels_dir / "labels.npz", labels=labels_arr, objframe_idx_2_label_idx=np.array([0], dtype=np.int64))
        np.save(labels_dir / "timestamps_us.npy", np.array([last_frame_ts_us], dtype=np.int64))

        # 2. Event Representations
        ev_repr_dir = seq_dir / "event_representations_v2"
        ev_repr_name = f"stacked_histogram_dt={TS_STEP_EV_REPR_MS}_nbins={EV_REPR_NBINS}"
        out_repr_dir = ev_repr_dir / ev_repr_name
        out_repr_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamps for representations: from 0 to 5s with 50ms step
        repr_timestamps_us = np.arange(TS_STEP_EV_REPR_MS, int(WINDOW_SIZE_SEC * 1000) + 1, TS_STEP_EV_REPR_MS, dtype=np.int64) * 1000
        np.save(out_repr_dir / "timestamps_us.npy", repr_timestamps_us)

        # objframe_idx_2_repr_idx: our single labeled frame is at last_frame_ts_us
        # which corresponds to the last representation index
        objframe_idx_2_repr_idx = np.array([len(repr_timestamps_us) - 1], dtype=np.int64)
        np.save(out_repr_dir / "objframe_idx_2_repr_idx.npy", objframe_idx_2_repr_idx)

        # Construct H5
        ev_repr_obj = StackedHistogram(bins=EV_REPR_NBINS, height=WINDOW_SIZE_POINTS_DS, width=NUM_CHANNELS)
        shape = ev_repr_obj.get_shape()

        with H5Writer(out_repr_dir / "event_representations.h5", "data", shape, np.uint8) as writer:
            ev_x = torch.from_numpy(events['x'])
            ev_y = torch.from_numpy(events['y'])
            ev_p = torch.from_numpy(events['p'])
            ev_t = torch.from_numpy(events['t'])

            for ts_us in repr_timestamps_us:
                t_start = ts_us - TS_STEP_EV_REPR_MS * 1000
                t_end = ts_us

                mask = (ev_t >= t_start) & (ev_t < t_end)
                if torch.any(mask):
                    repr_tensor = ev_repr_obj.construct(ev_x[mask], ev_y[mask], ev_p[mask], ev_t[mask])
                    writer.add_data(repr_tensor.numpy())
                else:
                    writer.add_data(np.zeros(shape, dtype=np.uint8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", default="data/H5/*.h5")
    parser.add_argument("--output_dir", default="data/styolo_dataset")
    parser.add_argument("--threshold", type=float, default=95.0)
    parser.add_argument("--min_area", type=int, default=4)
    args = parser.parse_args()

    input_files = sorted(glob.glob(args.input_glob))
    if not input_files:
        print(f"No files found for {args.input_glob}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = DASSequenceProcessor(output_dir, threshold_percentile=args.threshold, min_area=args.min_area)

    # Split files into train/val/test
    n = len(input_files)
    train_files = input_files[:int(0.7*n)]
    val_files = input_files[int(0.7*n):int(0.9*n)]
    test_files = input_files[int(0.9*n):]

    for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        if not files: continue
        print(f"\nProcessing {split} split...")
        for f in tqdm(files):
            processor.process_file(Path(f), split)

    print(f"\nDone! Dataset generated at {args.output_dir}")

if __name__ == "__main__":
    main()
