import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2

def visualize_sequence(sequence_dir: str, output_path: str = None):
    seq_path = Path(sequence_dir)
    print(f"Visualizing sequence: {seq_path}")

    # 1. Load labels
    labels_file = seq_path / 'labels_v2' / 'labels.npz'
    if not labels_file.exists():
        print(f"Error: Labels file not found at {labels_file}")
        return

    label_data = np.load(str(labels_file))
    labels = label_data['labels']
    objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']

    # 2. Load event representations
    ev_repr_parent = seq_path / 'event_representations_v2'
    # Find the representation directory (there should be only one)
    repr_dirs = list(ev_repr_parent.glob('stacked_histogram*'))
    if not repr_dirs:
        print(f"Error: No event representation directory found in {ev_repr_parent}")
        return

    repr_dir = repr_dirs[0]
    ev_repr_file = repr_dir / 'event_representations.h5'
    objframe_idx_2_repr_idx = np.load(str(repr_dir / 'objframe_idx_2_repr_idx.npy'))

    with h5py.File(str(ev_repr_file), 'r') as f:
        data = f['data'][:]

    print(f"Loaded event representations with shape: {data.shape}")
    print(f"Number of labeled frames: {len(objframe_idx_2_label_idx)}")

    # 3. Plot a few frames
    num_frames_to_plot = min(3, len(objframe_idx_2_label_idx))
    fig, axes = plt.subplots(num_frames_to_plot, 1, figsize=(10, 4 * num_frames_to_plot))
    if num_frames_to_plot == 1:
        axes = [axes]

    for i in range(num_frames_to_plot):
        frame_idx = i * (len(objframe_idx_2_label_idx) // num_frames_to_plot) if num_frames_to_plot > 1 else 0
        repr_idx = objframe_idx_2_repr_idx[frame_idx]

        # Collapse bins/polarities for visualization: sum all channels
        # Shape is (C, H, W) where C = 2 * nbins
        img = np.sum(data[repr_idx], axis=0)

        ax = axes[i]
        im = ax.imshow(img, cmap='viridis', aspect='auto')
        ax.set_title(f"Frame {frame_idx} (Repr Index {repr_idx})")
        plt.colorbar(im, ax=ax)

        # Draw bounding boxes
        start_lbl_idx = objframe_idx_2_label_idx[frame_idx]
        end_lbl_idx = objframe_idx_2_label_idx[frame_idx + 1] if frame_idx + 1 < len(objframe_idx_2_label_idx) else len(labels)

        frame_labels = labels[start_lbl_idx:end_lbl_idx]
        for lbl in frame_labels:
            # lbl is (t, x, y, w, h, class_id, conf)
            x, y, w, h = lbl['x'], lbl['y'], lbl['w'], lbl['h']
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y-2, f"class {int(lbl['class_id'])}", color='red', fontsize=8, fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize generated DAS st-yolo sequence')
    parser.add_argument('sequence_dir', help='Path to the sequence directory (e.g. data/st-yolo-das/train/sequence_000000)')
    parser.add_argument('--output', default='das_visualization.png', help='Output plot path')

    args = parser.parse_args()
    visualize_sequence(args.sequence_dir, args.output)
