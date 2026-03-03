"""
DAS瀑布图转换为st-yolo格式（针对256x64分辨率优化）

数据格式：
- 空间轴: 50个通道 (补齐到64)
- 时间轴: 2.56秒窗口 (100Hz -> 256点)
- 原始采样率: 1kHz
- 降采样因子: 10
- 窗口滑动: 1000ms步长 (可调)
"""

# =================== 必须最先设置的环境变量 ===================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# =============================================================

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
from dataclasses import dataclass, field
from enum import Enum, auto
import sys
import torch
from datetime import datetime

# 导入st-yolo的表示方法
# 使用insert(0, ...) 确保优先从st-yolo目录导入utils，避免与根目录的utils冲突
# 修正路径以指向正确的位置
STYOLO_BASE_PATH = Path(__file__).parent / 'applications/dvsdetection/st-yolo'
if str(STYOLO_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(STYOLO_BASE_PATH))

from data.utils.representations import StackedHistogram
from utils.preprocessing import _blosc_opts


# ======== DAS数据参数 ========
ORIGINAL_SAMPLING_RATE_HZ = 1000
DOWNSAMPLED_SAMPLING_RATE_HZ = 100
DOWNSAMPLE_FACTOR = ORIGINAL_SAMPLING_RATE_HZ // DOWNSAMPLED_SAMPLING_RATE_HZ  # 10

# 设定窗口为256点，匹配2.56秒
WINDOW_SIZE_POINTS_DS = 256
WINDOW_SIZE_SEC = WINDOW_SIZE_POINTS_DS / DOWNSAMPLED_SAMPLING_RATE_HZ  # 2.56s

STEP_SIZE_MS = 1000
STEP_SIZE_POINTS_DS = int(STEP_SIZE_MS * DOWNSAMPLED_SAMPLING_RATE_HZ / 1000)  # 100点

NUM_CHANNELS_RAW = 50
NUM_CHANNELS_PADDED = 64  # 补齐到64，满足32的倍数

MAX_DURATION_SEC = 600.0 # 增大最大时长限制
MAX_DURATION_POINTS_ORIGINAL = int(MAX_DURATION_SEC * ORIGINAL_SAMPLING_RATE_HZ)

# 事件表示参数
TS_STEP_EV_REPR_MS = 50  # 50ms生成一个事件表示
EV_REPR_NBINS = 10  # StackHistogram的bin数量


class AggregationType(Enum):
    COUNT = auto()
    DURATION = auto()


aggregation_2_string = {
    AggregationType.DURATION: 'dt',
    AggregationType.COUNT: 'ne',
}


@dataclass
class EventWindowExtractionConf:
    method: AggregationType = AggregationType.DURATION
    value: int = TS_STEP_EV_REPR_MS


@dataclass
class StackedHistogramConf:
    name: str = 'stacked_histogram'
    nbins: int = EV_REPR_NBINS
    count_cutoff: Optional[int] = None
    event_window_extraction: EventWindowExtractionConf = field(default_factory=EventWindowExtractionConf)
    fastmode: bool = True


# =============================================================

class H5Writer:
    """HDF5写入器"""

    def __init__(self, outfile: Path, key: str, ev_repr_shape: Tuple, numpy_dtype: np.dtype):
        assert len(ev_repr_shape) == 3
        self.h5f = h5py.File(str(outfile), 'w')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        self.key = key
        self.numpy_dtype = numpy_dtype

        # 创建HDF5数据集
        maxshape = (None,) + ev_repr_shape
        chunkshape = (1,) + ev_repr_shape
        self.maxshape = maxshape
        self.h5f.create_dataset(key, dtype=self.numpy_dtype.name, shape=chunkshape, chunks=chunkshape,
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


class DASSample:
    """表示一个DAS样本（2.56秒窗口，降采样后256点）"""

    def __init__(self, sample_id: int, start_time_idx_ds: int,
                 das_data_ds: np.ndarray, global_start_time_us: int):
        self.sample_id = sample_id
        self.start_time_idx_ds = start_time_idx_ds

        # 提取窗口数据 (T, W_raw)
        end_idx = min(start_time_idx_ds + WINDOW_SIZE_POINTS_DS, len(das_data_ds))
        raw_window = das_data_ds[start_time_idx_ds:end_idx].copy()

        # 时间轴填充 (T=256)
        if len(raw_window) < WINDOW_SIZE_POINTS_DS:
            pad_len = WINDOW_SIZE_POINTS_DS - len(raw_window)
            raw_window = np.vstack([raw_window, np.zeros((pad_len, NUM_CHANNELS_RAW), dtype=np.float32)])

        # 空间轴填充 (W=64)
        self.data = np.zeros((WINDOW_SIZE_POINTS_DS, NUM_CHANNELS_PADDED), dtype=np.float32)
        self.data[:, :NUM_CHANNELS_RAW] = raw_window

        # 计算全局时间戳
        self.start_time_us = global_start_time_us + (start_time_idx_ds * 10000)  # 100Hz = 10ms per sample
        self.end_time_us = self.start_time_us + int(WINDOW_SIZE_SEC * 1000000)

        self.height = WINDOW_SIZE_POINTS_DS  # 256
        self.width = NUM_CHANNELS_PADDED  # 64

    def detect_anomalies(self, threshold_percentile: float = 95.0):
        """检测异常（仅在有效通道上计算阈值）"""
        valid_data = self.data[:, :NUM_CHANNELS_RAW]
        threshold = np.percentile(valid_data, threshold_percentile)
        binary_mask = self.data > threshold
        return np.ascontiguousarray(binary_mask), threshold

    def convert_to_events(self, binary_mask: np.ndarray) -> Dict[str, np.ndarray]:
        y_idx, x_idx = np.where(binary_mask)

        # 计算时间戳（相对于窗口起始时间）
        relative_time_us = y_idx * 10000  # 100Hz = 10000μs per sample
        absolute_time_us = self.start_time_us + relative_time_us

        return {
            'x': x_idx.astype(np.int64),
            'y': y_idx.astype(np.int64),
            'p': np.ones_like(y_idx, dtype=np.int64),
            't': absolute_time_us.astype(np.int64)
        }

    def generate_labels(self, binary_mask: np.ndarray,
                       min_bbox_area: int = 4,
                       max_bbox_area: int = 2000) -> np.ndarray:
        labels = []
        if not np.any(binary_mask):
            return np.array([], dtype=[
                ('t', 'i8'), ('x', 'f4'), ('y', 'f4'), ('w', 'f4'),
                ('h', 'f4'), ('class_id', 'i4'), ('class_confidence', 'f4')
            ])

        mask_u8 = binary_mask.astype(np.uint8)
        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        # 对于 ST-YOLO，我们将标签分配给窗口的最后一个时间戳
        label_time_us = self.end_time_us

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_bbox_area or area > max_bbox_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            labels.append((label_time_us, float(x), float(y), float(w), float(h), 0, 1.0))

        dtype = np.dtype([
            ('t', 'i8'), ('x', 'f4'), ('y', 'f4'), ('w', 'f4'),
            ('h', 'f4'), ('class_id', 'i4'), ('class_confidence', 'f4')
        ])

        return np.array(labels, dtype=dtype) if labels else np.array([], dtype=dtype)


# =============================================================

class DASConverter:
    """DAS到st-yolo格式的转换器"""

    def __init__(self, input_h5_pattern: str, output_base_dir: str,
                 threshold_percentile: float = 95.0):
        self.input_pattern = input_h5_pattern
        self.output_dir = Path(output_base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_percentile = threshold_percentile

        print(f"参数配置:")
        print(f"  降采样率: {DOWNSAMPLED_SAMPLING_RATE_HZ}Hz")
        print(f"  窗口长度: {WINDOW_SIZE_SEC}秒 ({WINDOW_SIZE_POINTS_DS}点)")
        print(f"  输出分辨率: {WINDOW_SIZE_POINTS_DS}x{NUM_CHANNELS_PADDED}")

    def downsample_time_axis(self, data: np.ndarray) -> np.ndarray:
        H, W = data.shape
        new_H = H // DOWNSAMPLE_FACTOR
        trimmed_H = new_H * DOWNSAMPLE_FACTOR
        data = data[:trimmed_H]
        blocks = data.reshape(new_H, DOWNSAMPLE_FACTOR, W)
        downsampled = np.max(blocks, axis=1)
        return downsampled.astype(np.float32)

    def load_and_concat_h5_files(self) -> np.ndarray:
        files = sorted(glob.glob(self.input_pattern))
        assert files, f"未找到任何h5文件: {self.input_pattern}"

        all_data = []
        for f in tqdm(files, desc="读取h5文件"):
            with h5py.File(f, "r") as hf:
                keys = sorted(hf.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                # 如果没有数字key，尝试找 dataset
                if not keys:
                    blocks = [hf[k][:].copy() for k in hf.keys() if isinstance(hf[k], h5py.Dataset)]
                else:
                    blocks = [hf[k][:].copy() for k in keys]

                if not blocks: continue
                data = np.concatenate(blocks, axis=0).astype(np.float32)
                data = np.abs(data)
                all_data.append(data)

        concatenated = np.vstack(all_data)
        if len(concatenated) > MAX_DURATION_POINTS_ORIGINAL:
            concatenated = concatenated[:MAX_DURATION_POINTS_ORIGINAL]

        return self.downsample_time_axis(concatenated)

    def create_samples(self, data: np.ndarray) -> List[DASSample]:
        samples = []
        total = len(data)
        global_start_time_us = 0
        for start_idx in range(0, total - WINDOW_SIZE_POINTS_DS + 1, STEP_SIZE_POINTS_DS):
            sample = DASSample(
                sample_id=len(samples),
                start_time_idx_ds=start_idx,
                das_data_ds=data,
                global_start_time_us=global_start_time_us
            )
            samples.append(sample)
        return samples

    def save_labels(self, out_labels_dir: Path, labels: np.ndarray, timestamp_us: int):
        # ST-YOLO labels.npz 格式要求: objframe_idx_2_label_idx 长度应为 N_frames + 1
        np.savez(out_labels_dir / 'labels.npz',
                 labels=labels,
                 objframe_idx_2_label_idx=np.array([0, len(labels)], dtype=np.int64))
        np.save(out_labels_dir / 'timestamps_us.npy', np.array([timestamp_us], dtype=np.int64))

    def process_sample(self, sample: DASSample, sequence_dir: Path) -> bool:
        binary_mask, _ = sample.detect_anomalies(self.threshold_percentile)
        labels = sample.generate_labels(binary_mask)

        # 只保留有标签的样本，以提高训练效率并避免 ST-YOLO 报错
        if len(labels) == 0:
            return False

        sequence_name = f"sequence_{sample.sample_id:06d}"
        seq_path = sequence_dir / sequence_name
        seq_path.mkdir(parents=True, exist_ok=True)

        events = sample.convert_to_events(binary_mask)

        # 1. 保存标签
        labels_dir = seq_path / 'labels_v2'
        labels_dir.mkdir(parents=True, exist_ok=True)
        self.save_labels(labels_dir, labels, sample.end_time_us)

        # 2. 生成事件表示
        ev_repr_dir = seq_path / 'event_representations_v2'
        extraction_conf = EventWindowExtractionConf(method=AggregationType.DURATION, value=TS_STEP_EV_REPR_MS)
        ev_repr_name = f'stacked_histogram_{aggregation_2_string[extraction_conf.method]}={extraction_conf.value}_nbins={EV_REPR_NBINS}'
        out_repr_dir = ev_repr_dir / ev_repr_name
        out_repr_dir.mkdir(parents=True, exist_ok=True)

        # 生成表示时间轴：从 start + step 到 end，确保覆盖整个窗口
        # 2.56s / 50ms = 51.2 -> 51个完整bin
        repr_timestamps_us = np.arange(sample.start_time_us + TS_STEP_EV_REPR_MS*1000,
                                       sample.end_time_us,
                                       TS_STEP_EV_REPR_MS*1000, dtype=np.int64)
        # 强制最后一个表示的时间戳与标签时间戳对齐
        repr_timestamps_us = np.append(repr_timestamps_us, sample.end_time_us)
        np.save(out_repr_dir / 'timestamps_us.npy', repr_timestamps_us)

        # 映射标注帧到最后一个表示索引
        objframe_idx_2_repr_idx = np.array([len(repr_timestamps_us) - 1], dtype=np.int64)
        np.save(out_repr_dir / 'objframe_idx_2_repr_idx.npy', objframe_idx_2_repr_idx)

        ev_repr = StackedHistogram(bins=EV_REPR_NBINS, height=WINDOW_SIZE_POINTS_DS, width=NUM_CHANNELS_PADDED)
        shape = ev_repr.get_shape()

        with H5Writer(out_repr_dir / 'event_representations.h5', 'data', shape, ev_repr.get_numpy_dtype()) as h5_writer:
            ev_t = events['t']
            ev_x = torch.from_numpy(events['x'])
            ev_y = torch.from_numpy(events['y'])
            ev_p = torch.from_numpy(events['p'])
            ev_ts = torch.from_numpy(ev_t)

            for ts_end_us in repr_timestamps_us:
                ts_start_us = ts_end_us - TS_STEP_EV_REPR_MS*1000
                mask = (ev_ts >= ts_start_us) & (ev_ts < ts_end_us)

                if torch.any(mask):
                    repr_tensor = ev_repr.construct(ev_x[mask], ev_y[mask], ev_p[mask], ev_ts[mask])
                    h5_writer.add_data(repr_tensor.numpy())
                else:
                    h5_writer.add_data(np.zeros(shape, dtype=ev_repr.get_numpy_dtype()))

        return True

    def convert(self, train_ratio: float = 0.7, val_ratio: float = 0.2):
        data = self.load_and_concat_h5_files()
        samples = self.create_samples(data)

        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {'train': samples[:n_train], 'val': samples[n_train:n_train + n_val], 'test': samples[n_train + n_val:]}

        for split_name, split_samples in splits.items():
            print(f"\n处理{split_name}集 ({len(split_samples)}样本)...")
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            valid_count = 0
            for sample in tqdm(split_samples):
                if self.process_sample(sample, split_dir):
                    valid_count += 1
            print(f"  {split_name}集完成，共生成 {valid_count} 个有效序列")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pattern', default='data/H5/*.h5')
    parser.add_argument('--output_dir', default='data/st_yolo_das_v2')
    parser.add_argument('--threshold', type=float, default=95.0)
    args = parser.parse_args()

    converter = DASConverter(input_h5_pattern=args.input_pattern,
                             output_base_dir=args.output_dir,
                             threshold_percentile=args.threshold)
    converter.convert()


if __name__ == "__main__":
    main()
