import numpy as np
import h5py
import cv2
import hdf5plugin
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_sequence(sequence_dir: str):
    sequence_dir = Path(sequence_dir)

    # ========= 1. 读取事件表示 =========
    ev_dir = sequence_dir / "event_representations_v2"
    subdir = list(ev_dir.iterdir())[0]

    h5_file = subdir / "event_representations.h5"
    objframe_map_file = subdir / "objframe_idx_2_repr_idx.npy"

    with h5py.File(h5_file, "r") as f:
        ev_data = f["data"][:]   # shape: (80, 20, 400, 50)

    objframe_idx_2_repr_idx = np.load(objframe_map_file)
    repr_idx = objframe_idx_2_repr_idx[0]

    # 取最后一个repr
    repr_tensor = ev_data[repr_idx]   # (20, 400, 50)

    # ========= 2. 聚合通道用于可视化 =========
    # 把20个bin聚合成单通道
    repr_image = repr_tensor.sum(axis=0)   # (400, 50)

    # 归一化到 0~255
    repr_image = repr_image.astype(np.float32)
    repr_image -= repr_image.min()
    if repr_image.max() > 0:
        repr_image /= repr_image.max()
    repr_image = (repr_image * 255).astype(np.uint8)

    # 转成3通道方便画框
    vis_img = cv2.cvtColor(repr_image, cv2.COLOR_GRAY2BGR)

    # ========= 3. 读取labels =========
    label_file = sequence_dir / "labels_v2" / "labels.npz"
    label_data = np.load(label_file)

    labels = label_data["labels"]

    for label in labels:
        x = int(label["x"])
        y = int(label["y"])
        w = int(label["w"])
        h = int(label["h"])

        cv2.rectangle(
            vis_img,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            1
        )
        # cv2.putText(
        #     vis_img,
        #     str(label["class_id"]),
        #     (x, y - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.4,
        #     (0, 255, 0),
        #     1
        # )

    # ========= 4. 显示 =========
    plt.figure(figsize=(6, 10))
    plt.imshow(vis_img[:, :, ::-1], aspect="auto")  
    plt.title(sequence_dir.name)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    visualize_sequence("data/st_yolo_das/test/sequence_000823")