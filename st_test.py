"""
DAS瀑布图转换为st-yolo格式（优化版本）

数据格式：
- 空间轴: 50个通道
- 时间窗口: 4秒 (400点 @ 100Hz)
- 原始采样率: 1kHz，降采样到100Hz
- 滑动步长: 100ms (10点 @ 100Hz)
- 逐个处理h5文件

输出目录结构:
output_dir/
├── train/
│   ├── sequence_000000/
│   │   ├── labels_v2/
│   │   │   ├── labels.npz
│   │   │   └── timestamps_us.npy
│   │   └── event_representations_v2/
│   │       └── stacked_histogram_dt=50_nbins=10/
│   │           ├── event_representations.h5
│   │           ├── objframe_idx_2_repr_idx.npy
│   │           └── timestamps_us.npy
│   └── sequence_000001/
│       └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── dataset_info.json
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
from typing import List, Tuple, Dict, Optional
import argparse
from tqdm import tqdm
import glob
import weakref
import cv2
from dataclasses import dataclass, field
from enum import Enum, auto
import sys
import json
import torch
from datetime import datetime

# 导入st-yolo的表示方法
sys.path.insert(0, str(Path(__file__).parent / 'applications/dvsdetection/st-yolo'))
from data.utils.representations import StackedHistogram
from utils.preprocessing import _blosc_opts


# ======== DAS数据参数 ========
ORIGINAL_SAMPLING_RATE_HZ = 1000      # 原始采样率: 1kHz
DOWNSAMPLED_SAMPLING_RATE_HZ = 100    # 降采样率: 100Hz
DOWNSAMPLE_FACTOR = ORIGINAL_SAMPLING_RATE_HZ // DOWNSAMPLED_SAMPLING_RATE_HZ  # 10

# 窗口参数: 4秒窗口
WINDOW_SIZE_SEC = 4.0
WINDOW_SIZE_POINTS_DS = int(WINDOW_SIZE_SEC * DOWNSAMPLED_SAMPLING_RATE_HZ)  # 400点
WINDOW_SIZE_POINTS_ORIGINAL = int(WINDOW_SIZE_SEC * ORIGINAL_SAMPLING_RATE_HZ)  # 4000点

# 滑动步长: 100ms
STEP_SIZE_MS = 100
STEP_SIZE_POINTS_DS = int(STEP_SIZE_MS * DOWNSAMPLED_SAMPLING_RATE_HZ / 1000)  # 10点

# 空间维度
NUM_CHANNELS = 50

# 事件表示参数
TS_STEP_EV_REPR_MS = 50    # 每50ms生成一个事件表示
EV_REPR_NBINS = 10         # StackedHistogram的bin数量


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

    def get_current_length(self):
        return self.t_idx

    def add_data(self, data: np.ndarray):
        assert data.dtype == self.numpy_dtype
        assert data.shape == self.maxshape[1:]
        new_size = self.t_idx + 1
        self.h5f[self.key].resize(new_size, axis=0)
        self.h5f[self.key][self.t_idx:new_size] = data
        self.t_idx = new_size


# =============================================================

class DASSample:
    """表示一个DAS样本（4秒窗口，降采样后400点）"""
    
    def __init__(self, sample_id: int, start_time_idx_ds: int, 
                 das_data_ds: np.ndarray, global_start_time_us: int):
        """
        Args:
            sample_id: 样本ID（全局唯一）
            start_time_idx_ds: 在降采样数据中的起始索引
            das_data_ds: 完整的降采样后DAS数据
            global_start_time_us: 全局起始时间戳（微秒）
        """
        self.sample_id = sample_id
        self.start_time_idx_ds = start_time_idx_ds
        
        # 提取窗口数据
        end_idx = min(start_time_idx_ds + WINDOW_SIZE_POINTS_DS, len(das_data_ds))
        self.data = das_data_ds[start_time_idx_ds:end_idx].copy()
        
        # 如果窗口不足，进行填充
        if len(self.data) < WINDOW_SIZE_POINTS_DS:
            pad_len = WINDOW_SIZE_POINTS_DS - len(self.data)
            self.data = np.vstack([self.data, np.zeros((pad_len, self.data.shape[1]), dtype=np.float32)])
        
        # 时间戳（微秒）
        # 降采样后的每个点代表10ms (10000μs)
        self.start_time_us = global_start_time_us + int(start_time_idx_ds * 10000)
        self.end_time_us = self.start_time_us + int(WINDOW_SIZE_SEC * 1000000)
        
        self.height = WINDOW_SIZE_POINTS_DS  # 400
        self.width = NUM_CHANNELS  # 50
    
    def detect_anomalies(self, threshold_percentile: float = 99.0) -> Tuple[np.ndarray, float]:
        """检测异常区域"""
        threshold = np.percentile(self.data, threshold_percentile)
        binary_mask = self.data > threshold
        return np.ascontiguousarray(binary_mask), threshold
    
    def convert_to_events(self, binary_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        将二进制掩码转换为事件流格式
        
        返回: {'x': np.ndarray, 'y': np.ndarray, 'p': np.ndarray, 't': np.ndarray}
        - x: 空间坐标（通道索引）
        - y: 时间坐标（降采样后的时间索引）
        - p: 极性（DAS无极性，统一设为1）
        - t: 绝对时间戳（微秒）
        """
        y_idx, x_idx = np.where(binary_mask)
        
        if len(y_idx) == 0:
            return {
                'x': np.array([], dtype=np.int64),
                'y': np.array([], dtype=np.int64),
                'p': np.array([], dtype=np.int64),
                't': np.array([], dtype=np.int64)
            }
        
        # 计算绝对时间戳
        # y_idx是降采样后的索引，每个点对应10ms
        relative_time_us = y_idx * 10000  # 10ms = 10000μs
        absolute_time_us = self.start_time_us + relative_time_us
        
        # 极性：DAS数据统一设为1
        polarity = np.ones_like(y_idx, dtype=np.int64)
        
        return {
            'x': x_idx.astype(np.int64),
            'y': y_idx.astype(np.int64),
            'p': polarity,
            't': absolute_time_us.astype(np.int64)
        }
    
    def generate_labels(self, binary_mask: np.ndarray, 
                       min_bbox_area: int = 50,
                       max_bbox_area: int = 1000) -> np.ndarray:
        """
        生成标注标签
        
        返回: 结构化数组，包含 t, x, y, w, h, class_id, class_confidence
        """
        labels = []
        
        if not np.any(binary_mask):
            return np.array([], dtype=[
                ('t', 'i8'), ('x', 'f4'), ('y', 'f4'), ('w', 'f4'), 
                ('h', 'f4'), ('class_id', 'i4'), ('class_confidence', 'f4')
            ])
        
        mask_u8 = binary_mask.astype(np.uint8)
        
        # 使用连通区域分析
        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        
        for i in range(1, num_labels):  # 跳过背景（0）
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_bbox_area or area > max_bbox_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 确保最小尺寸
            w = max(w, 1)
            h = max(h, 1)
            
            # Clamp到边界，确保边界框完全在帧内
            # 关键：需要确保 x + w <= width 和 y + h <= height
            # 而不是简单地将 x, y clamp 到边界
            
            # 处理右边界
            if x + w > self.width:
                w = self.width - x
            # 处理下边界  
            if y + h > self.height:
                h = self.height - y
            
            # 处理左/上边界（超出帧的情况）
            if x < 0:
                w = w + x  # 减少宽度
                x = 0
            if y < 0:
                h = h + y  # 减少高度
                y = 0
            
            # 最终检查：确保边界框有效
            # 注意：y + h 必须 <= height，否则在 clamp_to_frame_ 中会出问题
            if w <= 0 or h <= 0:
                continue
            if x >= self.width or y >= self.height:
                continue
            # 额外保护：确保 y + h <= height
            if y + h > self.height:
                h = self.height - y
            if x + w > self.width:
                w = self.width - x
            if w <= 0 or h <= 0:
                continue
            
            # 所有边界框使用相同的时间戳（窗口末尾时间）
            # 这是st-yolo数据格式的要求：同一个帧的所有标签必须有相同的时间戳
            bbox_time_us = self.end_time_us
            
            labels.append((
                bbox_time_us,
                float(x),
                float(y),
                float(w),
                float(h),
                0,      # class_id: 0表示敲击事件
                1.0     # class_confidence
            ))

        dtype = np.dtype([
            ('t', 'i8'), ('x', 'f4'), ('y', 'f4'), ('w', 'f4'),
            ('h', 'f4'), ('class_id', 'i4'), ('class_confidence', 'f4')
        ])
        
        return np.array(labels, dtype=dtype) if labels else np.array([], dtype=dtype)


# =============================================================

class DASConverter:
    """DAS到st-yolo格式的转换器（逐文件处理）"""
    
    def __init__(self, input_h5_pattern: str, output_base_dir: str,
                 threshold_percentile: float = 97.0,
                 min_bbox_area: int = 50,
                 max_bbox_area: int = 1000):
        self.input_pattern = input_h5_pattern
        self.output_dir = Path(output_base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.threshold_percentile = threshold_percentile
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area = max_bbox_area
        
        # 事件表示名称
        self.ev_repr_name = f'stacked_histogram_dt={TS_STEP_EV_REPR_MS}_nbins={EV_REPR_NBINS}'
        
        self._print_config()
    
    def _print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("DAS-ST-YOLO 数据集生成配置")
        print("=" * 60)
        print(f"原始采样率: {ORIGINAL_SAMPLING_RATE_HZ}Hz")
        print(f"降采样率: {DOWNSAMPLED_SAMPLING_RATE_HZ}Hz")
        print(f"降采样因子: {DOWNSAMPLE_FACTOR}")
        print(f"窗口长度: {WINDOW_SIZE_SEC}秒 ({WINDOW_SIZE_POINTS_DS}点)")
        print(f"滑动步长: {STEP_SIZE_MS}ms ({STEP_SIZE_POINTS_DS}点)")
        print(f"通道数: {NUM_CHANNELS}")
        print(f"事件表示步长: {TS_STEP_EV_REPR_MS}ms")
        print(f"StackHistogram bins: {EV_REPR_NBINS}")
        print(f"异常检测阈值百分位: {self.threshold_percentile}")
        print(f"边界框面积范围: [{self.min_bbox_area}, {self.max_bbox_area}]")
        print(f"输出分辨率: {WINDOW_SIZE_POINTS_DS} × {NUM_CHANNELS}")
        print("=" * 60)
    
    def downsample_time_axis(self, data: np.ndarray) -> np.ndarray:
        """
        时间轴降采样（使用块最大值聚合，适合异常检测）
        
        从1kHz降采样到100Hz (降采样因子=10)
        """
        H, W = data.shape
        new_H = H // DOWNSAMPLE_FACTOR
        
        # 确保能整除
        trimmed_H = new_H * DOWNSAMPLE_FACTOR
        data = data[:trimmed_H]
        
        # 使用最大值聚合（适合异常检测，保留峰值）
        blocks = data.reshape(new_H, DOWNSAMPLE_FACTOR, W)
        downsampled = np.max(blocks, axis=1)
        
        return downsampled.astype(np.float32)
    
    def load_h5_file(self, h5_path: str) -> np.ndarray:
        """加载单个H5文件并降采样"""
        print(f"  加载文件: {h5_path}")
        
        with h5py.File(h5_path, "r") as hf:
            keys = sorted(hf.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            blocks = [hf[k][:].copy() for k in keys]
            data = np.concatenate(blocks, axis=0).astype(np.float32)
            data = np.abs(data)  # 取绝对值
        
        print(f"    原始数据形状: {data.shape}")
        
        # 时间轴降采样
        downsampled = self.downsample_time_axis(data)
        print(f"    降采样后形状: {downsampled.shape}")
        
        return downsampled
    
    def create_samples_from_file(self, data_ds: np.ndarray, 
                                  start_sample_id: int,
                                  file_start_time_us: int) -> List[DASSample]:
        """
        从单个文件的降采样数据创建滑动窗口样本
        
        Args:
            data_ds: 降采样后的数据
            start_sample_id: 起始样本ID
            file_start_time_us: 文件起始时间戳（微秒）
        """
        samples = []
        total = len(data_ds)
        
        # 滑动窗口
        sample_id = start_sample_id
        for start_idx in range(0, total - WINDOW_SIZE_POINTS_DS + 1, STEP_SIZE_POINTS_DS):
            sample = DASSample(
                sample_id=sample_id,
                start_time_idx_ds=start_idx,
                das_data_ds=data_ds,
                global_start_time_us=file_start_time_us
            )
            samples.append(sample)
            sample_id += 1
        
        return samples
    
    def generate_ev_repr_timestamps(self, sample: DASSample) -> np.ndarray:
        """
        生成事件表示的时间戳序列
        
        每TS_STEP_EV_REPR_MS(50ms)生成一个事件表示
        4秒窗口 = 4000ms / 50ms = 80个事件表示
        """
        ev_repr_timestamps = []
        current_ts = sample.start_time_us
        
        while current_ts <= sample.end_time_us:
            ev_repr_timestamps.append(current_ts)
            current_ts += TS_STEP_EV_REPR_MS * 1000  # 转换为微秒
        
        return np.array(ev_repr_timestamps, dtype=np.int64)
    
    def save_labels(self, out_labels_dir: Path, labels: np.ndarray, 
                   frame_timestamps_us: np.ndarray):
        """
        保存标签文件
        
        目录结构:
        labels_v2/
        ├── labels.npz  (包含labels和objframe_idx_2_label_idx)
        └── timestamps_us.npy
        """
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # objframe_idx_2_label_idx: 每个标签帧的起始索引
        # 当 labels 为空时，objframe_idx_2_label_idx 也应该是空数组
        if len(labels) == 0:
            objframe_idx_2_label_idx = np.array([], dtype=np.int64)
        else:
            # 每个样本只有一个标签帧，索引从0开始
            objframe_idx_2_label_idx = np.array([0], dtype=np.int64)
        
        # 保存labels.npz
        outfile_labels = out_labels_dir / 'labels.npz'
        np.savez(str(outfile_labels), 
                 labels=labels, 
                 objframe_idx_2_label_idx=objframe_idx_2_label_idx)
        
        # 保存timestamps_us.npy
        out_labels_ts_file = out_labels_dir / 'timestamps_us.npy'
        np.save(str(out_labels_ts_file), frame_timestamps_us)
    
    def write_event_representations(self, events: Dict[str, np.ndarray],
                                   ev_repr_timestamps_us: np.ndarray,
                                   ev_out_dir: Path):
        """
        写入事件表示
        
        目录结构:
        event_representations_v2/
        └── stacked_histogram_dt=50_nbins=10/
            ├── event_representations.h5
            ├── objframe_idx_2_repr_idx.npy
            └── timestamps_us.npy
        """
        ev_out_dir = ev_out_dir / self.ev_repr_name
        ev_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存timestamps_us.npy
        timestamps_file = ev_out_dir / 'timestamps_us.npy'
        np.save(str(timestamps_file), ev_repr_timestamps_us)
        
        # 创建事件表示器
        ev_repr = StackedHistogram(
            bins=EV_REPR_NBINS,
            height=WINDOW_SIZE_POINTS_DS,
            width=NUM_CHANNELS,
            count_cutoff=None,
            fastmode=True
        )
        
        ev_repr_shape = tuple(ev_repr.get_shape())  # (2*nbins, height, width) = (20, 400, 50)
        ev_repr_dtype = ev_repr.get_numpy_dtype()  # uint8
        
        outfile = ev_out_dir / 'event_representations.h5'
        
        # 提取事件数据
        ev_t = events['t']
        ev_x = events['x']
        ev_y = events['y']
        ev_p = events['p']
        
        with H5Writer(outfile, key='data', ev_repr_shape=ev_repr_shape, 
                      numpy_dtype=ev_repr_dtype) as h5_writer:
            
            # 对每个事件表示时间窗口生成表示
            for ev_repr_ts in ev_repr_timestamps_us:
                # 提取时间窗口内的事件 [ts-50ms, ts]
                ts_start_us = ev_repr_ts - (TS_STEP_EV_REPR_MS * 1000)
                ts_end_us = ev_repr_ts
                
                # 二分查找事件索引
                start_idx = np.searchsorted(ev_t, ts_start_us, side='left')
                end_idx = np.searchsorted(ev_t, ts_end_us, side='right')
                
                # 提取事件数据
                window_x = ev_x[start_idx:end_idx]
                window_y = ev_y[start_idx:end_idx]
                window_p = ev_p[start_idx:end_idx]
                window_t = ev_t[start_idx:end_idx]
                
                # 构造事件表示
                if len(window_x) > 0:
                    # 需要将时间戳归一化到窗口内
                    # StackedHistogram.construct需要时间戳在窗口范围内
                    t0 = ts_start_us
                    t1 = ts_end_us
                    window_t_normalized = window_t - t0  # 相对于窗口起始的时间
                    
                    ev_repr_tensor = ev_repr.construct(
                        torch.from_numpy(window_x),
                        torch.from_numpy(window_y),
                        torch.from_numpy(window_p),
                        torch.from_numpy(window_t_normalized)
                    )
                    ev_repr_numpy = ev_repr_tensor.numpy()
                else:
                    # 空窗口，创建全零表示
                    ev_repr_numpy = np.zeros(ev_repr_shape, dtype=ev_repr_dtype)
                
                h5_writer.add_data(ev_repr_numpy)
        
        # 生成objframe_idx_2_repr_idx映射
        # 每个样本只有一个标签帧，对应最后一个事件表示
        # 4秒窗口 / 50ms = 80个事件表示，索引从0开始
        num_ev_reprs = len(ev_repr_timestamps_us)
        objframe_idx_2_repr_idx = np.array([num_ev_reprs - 1], dtype=np.int64)
        np.save(str(ev_out_dir / 'objframe_idx_2_repr_idx.npy'), objframe_idx_2_repr_idx)
    
    def process_sample(self, sample: DASSample, sequence_dir: Path) -> Optional[Dict]:
        """
        处理单个样本
        
        返回:
            成功时返回样本信息字典，没有标签时返回None
        
        目录结构:
        sequence_XXXXXX/
        ├── labels_v2/
        │   ├── labels.npz
        │   └── timestamps_us.npy
        └── event_representations_v2/
            └── stacked_histogram_dt=50_nbins=10/
                ├── event_representations.h5
                ├── objframe_idx_2_repr_idx.npy
                └── timestamps_us.npy
        """
        # 1. 检测异常
        binary_mask, threshold = sample.detect_anomalies(self.threshold_percentile)
        
        # 2. 转换为事件
        events = sample.convert_to_events(binary_mask)
        
        # 3. 生成标签
        labels = sample.generate_labels(binary_mask, self.min_bbox_area, self.max_bbox_area)
        
        # 如果没有标签，跳过该样本（对于检测任务，空标签样本没有意义）
        if len(labels) == 0:
            return None
        
        # 创建序列目录
        sequence_name = f"sequence_{sample.sample_id:06d}"
        sequence_dir = sequence_dir / sequence_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. 标签时间戳（窗口末尾时间）
        frame_timestamps_us = np.array([sample.end_time_us], dtype=np.int64)
        
        # 5. 保存标签
        labels_dir = sequence_dir / 'labels_v2'
        self.save_labels(labels_dir, labels, frame_timestamps_us)
        
        # 6. 生成事件表示时间戳
        ev_repr_timestamps_us = self.generate_ev_repr_timestamps(sample)
        
        # 7. 写入事件表示
        ev_repr_dir = sequence_dir / 'event_representations_v2'
        self.write_event_representations(events, ev_repr_timestamps_us, ev_repr_dir)
        
        # 返回样本信息
        return {
            'sample_id': sample.sample_id,
            'start_time_us': sample.start_time_us,
            'end_time_us': sample.end_time_us,
            'num_events': len(events['t']),
            'num_labels': len(labels),
            'threshold': float(threshold)
        }
    
    def convert(self, train_ratio: float = 0.7, val_ratio: float = 0.2):
        """
        完整转换流程：逐文件处理
        """
        # 获取所有h5文件
        h5_files = sorted(glob.glob(self.input_pattern))
        assert h5_files, f"未找到任何h5文件: {self.input_pattern}"
        
        print(f"\n找到 {len(h5_files)} 个H5文件:")
        for f in h5_files:
            print(f"  - {f}")
        
        # 逐文件处理，收集所有样本
        all_samples = []
        current_sample_id = 0
        current_time_us = 0  # 全局时间累计（微秒）
        
        for h5_file in h5_files:
            print(f"\n处理文件: {h5_file}")
            
            # 加载并降采样
            data_ds = self.load_h5_file(h5_file)
            
            # 计算该文件的时长（微秒）
            file_duration_us = len(data_ds) * 10000  # 每个降采样点10ms
            
            # 创建样本
            samples = self.create_samples_from_file(
                data_ds, 
                start_sample_id=current_sample_id,
                file_start_time_us=current_time_us
            )
            
            print(f"  生成 {len(samples)} 个样本")
            
            all_samples.extend(samples)
            current_sample_id += len(samples)
            current_time_us += file_duration_us
        
        print(f"\n总共生成 {len(all_samples)} 个样本")
        print(f"总时长: {current_time_us / 1000000:.1f} 秒")
        
        # 划分数据集（按样本顺序）
        n = len(all_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            'train': all_samples[:n_train],
            'val': all_samples[n_train:n_train + n_val],
            'test': all_samples[n_train + n_val:]
        }
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(splits['train'])} 样本")
        print(f"  验证集: {len(splits['val'])} 样本")
        print(f"  测试集: {len(splits['test'])} 样本")
        
        # 处理每个样本
        sample_infos = {'train': [], 'val': [], 'test': []}
        skipped_counts = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, split_samples in splits.items():
            if not split_samples:
                continue
                
            print(f"\n处理 {split_name} 集...")
            split_out_dir = self.output_dir / split_name
            split_out_dir.mkdir(parents=True, exist_ok=True)
            
            for sample in tqdm(split_samples, desc=f"  处理{split_name}"):
                info = self.process_sample(sample, Path(split_out_dir))
                if info is not None:
                    sample_infos[split_name].append(info)
                else:
                    skipped_counts[split_name] += 1
        
        # 打印跳过统计
        print(f"\n跳过样本统计（无标签）:")
        for split_name, count in skipped_counts.items():
            total = len(splits[split_name])
            print(f"  {split_name}: {count}/{total} 样本被跳过")
        
        # 打印实际保存的样本数量
        print(f"\n实际保存的样本数量:")
        for split_name, infos in sample_infos.items():
            print(f"  {split_name}: {len(infos)} 个有效样本")
        
        # 生成数据集元信息
        self.generate_dataset_info(sample_infos)
        
        print(f"\n转换完成！")
        print(f"输出目录: {self.output_dir}")
    
    def generate_dataset_info(self, sample_infos: Dict):
        """生成数据集元信息文件"""
        dataset_info = {
            "dataset_name": "DAS-ST-YOLO",
            "creation_time": datetime.now().isoformat(),
            "parameters": {
                "original_sampling_rate_hz": ORIGINAL_SAMPLING_RATE_HZ,
                "downsampled_sampling_rate_hz": DOWNSAMPLED_SAMPLING_RATE_HZ,
                "downsample_factor": DOWNSAMPLE_FACTOR,
                "window_size_sec": WINDOW_SIZE_SEC,
                "window_size_points_ds": WINDOW_SIZE_POINTS_DS,
                "step_size_ms": STEP_SIZE_MS,
                "step_size_points_ds": STEP_SIZE_POINTS_DS,
                "num_channels": NUM_CHANNELS,
                "resolution_hw": [WINDOW_SIZE_POINTS_DS, NUM_CHANNELS],
                "ts_step_ev_repr_ms": TS_STEP_EV_REPR_MS,
                "ev_repr_nbins": EV_REPR_NBINS,
                "ev_repr_name": self.ev_repr_name,
                "threshold_percentile": self.threshold_percentile,
                "min_bbox_area": self.min_bbox_area,
                "max_bbox_area": self.max_bbox_area
            },
            "splits": {
                split_name: {
                    "count": len(infos),
                    "sequences": [f"sequence_{info['sample_id']:06d}" for info in infos]
                }
                for split_name, infos in sample_infos.items()
            },
            "total_sequences": sum(len(infos) for infos in sample_infos.values()),
            "sample_statistics": {
                split_name: {
                    "total_events": sum(info['num_events'] for info in infos),
                    "total_labels": sum(info['num_labels'] for info in infos),
                    "avg_events_per_sample": float(np.mean([info['num_events'] for info in infos])) if infos else 0,
                    "avg_labels_per_sample": float(np.mean([info['num_labels'] for info in infos])) if infos else 0
                }
                for split_name, infos in sample_infos.items()
            }
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n数据集信息已保存到: {info_file}")


# =============================================================

def main():
    parser = argparse.ArgumentParser(description='DAS瀑布图转st-yolo格式（优化版）')
    parser.add_argument('--input_pattern', default='data/H5/*.h5', 
                        help='输入H5文件模式（如: data/H5/*.h5）')
    parser.add_argument('--output_dir', default=f'data/st_yolo_{TS_STEP_EV_REPR_MS}ms_{EV_REPR_NBINS}bins', 
                        help='输出目录')
    parser.add_argument('--threshold_percentile', type=float, default=98.0, 
                        help='异常检测阈值百分位（默认97）')
    parser.add_argument('--min_bbox_area', type=int, default=30, 
                        help='最小边界框面积（默认20）')
    parser.add_argument('--max_bbox_area', type=int, default=1000, 
                        help='最大边界框面积（默认1000）')
    parser.add_argument('--train_ratio', type=float, default=0.7, 
                        help='训练集比例（默认0.7）')
    parser.add_argument('--val_ratio', type=float, default=0.2, 
                        help='验证集比例（默认0.2）')
    
    args = parser.parse_args()
    
    converter = DASConverter(
        input_h5_pattern=args.input_pattern,
        output_base_dir=args.output_dir,
        threshold_percentile=args.threshold_percentile,
        min_bbox_area=args.min_bbox_area,
        max_bbox_area=args.max_bbox_area
    )
    
    converter.convert(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )


if __name__ == "__main__":
    main()