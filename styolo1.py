"""
DAS瀑布图转换为st-yolo格式（参考preprocess_dataset.py完整流程）
 
数据格式：
- 空间轴: 50个通道
- 时间轴降采样: 200点/2秒窗口 (100Hz)
- 原始采样率: 1kHz
- 降采样因子: 10
- 窗口滑动: 100ms步长
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
sys.path.insert(0, str(Path(__file__).parent / 'applications/dvsdetection/st_yolo'))
from data.utils.representations import StackedHistogram, RepresentationBase
from utils.preprocessing import _blosc_opts
 
 
# ======== DAS数据参数 ========
# 原始: 1kHz采样率
# 降采样后: 100Hz (200点/2秒)
# 降采样因子: 10
ORIGINAL_SAMPLING_RATE_HZ = 1000
DOWNSAMPLED_SAMPLING_RATE_HZ = 100
DOWNSAMPLE_FACTOR = ORIGINAL_SAMPLING_RATE_HZ // DOWNSAMPLED_SAMPLING_RATE_HZ  # 10
 
WINDOW_SIZE_SEC = 2.0
WINDOW_SIZE_POINTS_DS = int(WINDOW_SIZE_SEC * DOWNSAMPLED_SAMPLING_RATE_HZ)  # 200点
WINDOW_SIZE_POINTS_ORIGINAL = int(WINDOW_SIZE_SEC * ORIGINAL_SAMPLING_RATE_HZ)  # 2000点
 
STEP_SIZE_MS = 100
STEP_SIZE_POINTS_DS = int(STEP_SIZE_MS * DOWNSAMPLED_SAMPLING_RATE_HZ / 1000)  # 10点
 
NUM_CHANNELS = 50

MAX_DURATION_POINTS_ORIGINAL = 30000  # 30000点
MAX_DURATION_SEC = 30.0
MAX_DURATION_POINTS_DS = int(MAX_DURATION_SEC * DOWNSAMPLED_SAMPLING_RATE_HZ)  # 3000点
 
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
    """HDF5写入器（参考preprocess_dataset.py）"""
    
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
 
    def get_current_length(self):
        return self.t_idx
 
    def add_data(self, data: np.ndarray):
        assert data.dtype == self.numpy_dtype
        assert data.shape == self.maxshape[1:]
        new_size = self.t_idx + 1
        self.h5f[self.key].resize(new_size, axis=0)
        self.h5f[self.key][self.t_idx:new_size] = data
        self.t_idx = new_size
 
 
class DASSample:
    """表示一个DAS样本（2秒窗口，降采样后200点）"""
    
    def __init__(self, sample_id: int, start_time_idx_ds: int, 
                 das_data_ds: np.ndarray, global_start_time_us: int):
        """
        Args:
            sample_id: 样本ID
            start_time_idx_ds: 降采样数据的起始索引
            das_data_ds: 降采样后的DAS数据
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
        
        # 计算全局时间戳
        self.start_time_us = global_start_time_us + (start_time_idx_ds * 10000)  # 100Hz = 10ms per sample
        self.end_time_us = self.start_time_us + (WINDOW_SIZE_SEC * 1000000)  # 微秒
        
        self.height = WINDOW_SIZE_POINTS_DS  # 200
        self.width = NUM_CHANNELS  # 50
    
    def detect_anomalies(self, threshold_percentile: float = 95.0):
        """检测异常"""
        threshold = np.percentile(self.data, threshold_percentile)
        binary_mask = self.data > threshold
        return np.ascontiguousarray(binary_mask), threshold
    
    def convert_to_events(self, binary_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        将二进制掩码转换为事件流
        
        返回: {'x': np.ndarray, 'y': np.ndarray, 'p': np.ndarray, 't': np.ndarray}
        """
        y_idx, x_idx = np.where(binary_mask)
        
        # 计算时间戳（相对于窗口起始时间）
        # y_idx是降采样后的索引，需要转换回原始时间戳
        original_time_idx = y_idx * DOWNSAMPLE_FACTOR
        relative_time_us = original_time_idx * 1000  # 1kHz = 1000μs per sample
        
        # 转换为绝对时间戳
        absolute_time_us = self.start_time_us + relative_time_us
        
        # 极性：DAS数据没有天然极性，统一设为1
        polarity = np.ones_like(y_idx, dtype=np.int64)
        
        return {
            'x': x_idx.astype(np.int64),
            'y': y_idx.astype(np.int64),
            'p': polarity,
            't': absolute_time_us.astype(np.int64)
        }
    
    def generate_labels(self, binary_mask: np.ndarray, 
                       min_bbox_area: int = 10,
                       max_bbox_area: int = 300) -> np.ndarray:
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
            
            # Clamp到边界
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            
            if x + w > self.width:
                w = self.width - x
            if y + h > self.height:
                h = self.height - y
            
            if w <= 0 or h <= 0:
                continue
            
            # 为了与st-yolo格式一致，同一帧内的所有标签应使用相同的起始时间戳
            bbox_time_us = self.start_time_us
            
            labels.append((
                bbox_time_us,
                float(x),
                float(y),
                float(w),
                float(h),
                0,      # class_id
                1.0     # class_confidence
            ))

        
        dtype = np.dtype([
            ('t', 'i8'), ('x', 'f4'), ('y', 'f4'), ('w', 'f4'),
            ('h', 'f4'), ('class_id', 'i4'), ('class_confidence', 'f4')
        ])
        
        return np.array(labels, dtype=dtype) if labels else np.array([], dtype=dtype)
 
 
# =============================================================
 
class DASConverter:
    """DAS到st-yolo格式的转换器（参考preprocess_dataset.py流程）"""
    
    def __init__(self, input_h5_pattern: str, output_base_dir: str,
                 threshold_percentile: float = 95.0):
        self.input_pattern = input_h5_pattern
        self.output_dir = Path(output_base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.threshold_percentile = threshold_percentile
        
        print(f"参数配置:")
        print(f"  原始采样率: {ORIGINAL_SAMPLING_RATE_HZ}Hz")
        print(f"  降采样率: {DOWNSAMPLED_SAMPLING_RATE_HZ}Hz")
        print(f"  降采样因子: {DOWNSAMPLE_FACTOR}")
        print(f"  窗口长度: {WINDOW_SIZE_SEC}秒 ({WINDOW_SIZE_POINTS_DS}点)")
        print(f"  滑动步长: {STEP_SIZE_MS}ms ({STEP_SIZE_POINTS_DS}点)")
        print(f"  最大时长: {MAX_DURATION_SEC}秒")
        print(f"  通道数: {NUM_CHANNELS}")
        print(f"  事件表示步长: {TS_STEP_EV_REPR_MS}ms")
        print(f"  StackHistogram bins: {EV_REPR_NBINS}")
    
    def downsample_time_axis(self, data: np.ndarray) -> np.ndarray:
        """
        时间轴降采样（使用块聚合，适合异常检测）
        
        从1kHz降采样到100Hz (降采样因子=10)
        """
        H, W = data.shape
        new_H = H // DOWNSAMPLE_FACTOR
        
        # 确保能整除
        trimmed_H = new_H * DOWNSAMPLE_FACTOR
        data = data[:trimmed_H]
        
        # 使用最大值聚合（适合异常检测）
        blocks = data.reshape(new_H, DOWNSAMPLE_FACTOR, W)
        downsampled = np.max(blocks, axis=1)
        
        return downsampled.astype(np.float32)
    
    def load_and_concat_h5_files(self) -> np.ndarray:
        """加载并拼接H5文件"""
        files = sorted(glob.glob(self.input_pattern))
        assert files, f"未找到任何h5文件: {self.input_pattern}"
        
        all_data = []
        
        for f in tqdm(files, desc="读取h5文件"):
            with h5py.File(f, "r") as hf:
                keys = sorted(hf.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                blocks = [hf[k][:].copy() for k in keys]
                data = np.concatenate(blocks, axis=0).astype(np.float32)
                data = np.abs(data)  # 取绝对值
                all_data.append(data)
        
        concatenated = np.vstack(all_data)
        
        # 限制在最大时长内
        if len(concatenated) > MAX_DURATION_POINTS_ORIGINAL:
            print(f"数据总长度: {len(concatenated)}点，限制为{MAX_DURATION_POINTS_ORIGINAL}点")
            concatenated = concatenated[:MAX_DURATION_POINTS_ORIGINAL]
        
        # 时间轴降采样
        print(f"降采样前: {concatenated.shape}")
        downsampled = self.downsample_time_axis(concatenated)
        print(f"降采样后: {downsampled.shape}")
        
        return downsampled
    
    def create_samples(self, data: np.ndarray) -> List[DASSample]:
        """创建滑动窗口样本"""
        samples = []
        total = len(data)
        
        print(f"创建样本: 总长度{total}点, 窗口{WINDOW_SIZE_POINTS_DS}点, 步长{STEP_SIZE_POINTS_DS}点")
        
        global_start_time_us = 0
        for start_idx in range(0, total - WINDOW_SIZE_POINTS_DS + 1, STEP_SIZE_POINTS_DS):
            sample = DASSample(
                sample_id=len(samples),
                start_time_idx_ds=start_idx,
                das_data_ds=data,
                global_start_time_us=global_start_time_us
            )
            samples.append(sample)
        
        print(f"总共创建 {len(samples)} 个样本")
        return samples
    
    def save_labels(self, out_labels_dir: Path, labels_per_frame: List[np.ndarray],
                   frame_timestamps_us: np.ndarray):
        """
        保存标签文件（参考preprocess_dataset.py的save_labels函数）
        
        目录结构:
        labels_v2/
        ├── labels.npz  (包含labels和objframe_idx_2_label_idx)
        └── timestamps_us.npy
        """
        assert len(labels_per_frame) == len(frame_timestamps_us)
        assert len(labels_per_frame) > 0
        
        labels_v2 = list()
        objframe_idx_2_label_idx = list()
        start_idx = 0
        
        for labels, timestamp in zip(labels_per_frame, frame_timestamps_us):
            objframe_idx_2_label_idx.append(start_idx)
            labels_v2.append(labels)
            start_idx += len(labels)
        
        assert len(labels_v2) == len(objframe_idx_2_label_idx)
        labels_v2 = np.concatenate(labels_v2)
        
        # 保存labels.npz
        outfile_labels = out_labels_dir / 'labels.npz'
        np.savez(str(outfile_labels), labels=labels_v2, objframe_idx_2_label_idx=objframe_idx_2_label_idx)
        
        # 保存timestamps_us.npy
        out_labels_ts_file = out_labels_dir / 'timestamps_us.npy'
        np.save(str(out_labels_ts_file), frame_timestamps_us)
    
    def generate_ev_repr_timestamps(self, sample_start_us: int, sample_end_us: int):
        """
        生成事件表示的时间戳序列（参考preprocess_dataset.py）
        
        按TS_STEP_EV_REPR_MS间隔生成时间戳
        """
        ev_repr_timestamps = []
        current_ts = sample_start_us
        
        while current_ts <= sample_end_us:
            ev_repr_timestamps.append(current_ts)
            current_ts += (TS_STEP_EV_REPR_MS * 1000)  # 转换为微秒
        
        return np.array(ev_repr_timestamps, dtype=np.int64)
    
    def write_event_representations(self, events: Dict[str, np.ndarray],
                                   ev_repr_timestamps_us: np.ndarray,
                                   ev_out_dir: Path):
        """
        写入事件表示（参考preprocess_dataset.py的write_event_representations）
        
        目录结构:
        event_representations_v2/
        └── stacked_histogram_dt=50_nbins=10/
            ├── event_representations.h5
            ├── objframe_idx_2_repr_idx.npy
            └── timestamps_us.npy
        """
        # 生成事件表示名称
        extraction_conf = EventWindowExtractionConf(method=AggregationType.DURATION, value=TS_STEP_EV_REPR_MS)
        ev_repr_name = f'stacked_histogram_{aggregation_2_string[extraction_conf.method]}={extraction_conf.value}_nbins={EV_REPR_NBINS}'
        
        ev_out_dir = ev_out_dir / ev_repr_name
        ev_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存timestamps_us.npy
        timestamps_file = ev_out_dir / 'timestamps_us.npy'
        np.save(str(timestamps_file), ev_repr_timestamps_us)
        
        # 创建事件表示
        ev_repr = StackedHistogram(
            bins=EV_REPR_NBINS,
            height=WINDOW_SIZE_POINTS_DS,
            width=NUM_CHANNELS,
            count_cutoff=None,
            fastmode=True
        )
        
        ev_repr_shape = tuple(ev_repr.get_shape())  # (2*nbins, height, width)
        ev_repr_dtype = ev_repr.get_numpy_dtype()
        
        outfile = ev_out_dir / 'event_representations.h5'
        
        with H5Writer(outfile, key='data', ev_repr_shape=ev_repr_shape, numpy_dtype=ev_repr_dtype) as h5_writer:
            ev_ts_us = events['t']
            x = events['x']
            y = events['y']
            p = events['p']
            
            # 对每个事件表示时间窗口生成表示
            for ev_repr_ts in ev_repr_timestamps_us:
                # 提取时间窗口内的事件
                ts_start_us = ev_repr_ts - (TS_STEP_EV_REPR_MS * 1000)
                ts_end_us = ev_repr_ts
                
                start_idx = np.searchsorted(ev_ts_us, ts_start_us, side='left')
                end_idx = np.searchsorted(ev_ts_us, ts_end_us, side='right')
                
                # 提取事件数据
                window_x = x[start_idx:end_idx]
                window_y = y[start_idx:end_idx]
                window_p = p[start_idx:end_idx]
                window_t = ev_ts_us[start_idx:end_idx]
                
                # 构造事件表示
                if len(window_x) > 0:
                    ev_repr_tensor = ev_repr.construct(
                        torch.from_numpy(window_x),
                        torch.from_numpy(window_y),
                        torch.from_numpy(window_p),
                        torch.from_numpy(window_t)
                    )
                    ev_repr_numpy = ev_repr_tensor.numpy()
                else:
                    # 空窗口，创建全零表示
                    ev_repr_numpy = np.zeros(ev_repr_shape, dtype=ev_repr_dtype)
                
                h5_writer.add_data(ev_repr_numpy)
        
        print(f"  事件表示保存到: {outfile}")
    
    def process_sample(self, sample: DASSample, sequence_dir: Path):
        """
        处理单个样本（参考preprocess_dataset.py的process_sequence）
        
        目录结构:
        sequence_00000X/
        ├── labels_v2/
        └── event_representations_v2/
        """
        # 创建序列目录
        sequence_name = f"sequence_{sample.sample_id:06d}"
        sequence_dir = sequence_dir / sequence_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 检测异常
        binary_mask, threshold = sample.detect_anomalies(self.threshold_percentile)
        
        # 2. 转换为事件
        events = sample.convert_to_events(binary_mask)
        
        # 3. 生成标签
        labels = sample.generate_labels(binary_mask)
        
        # 4. 准备标签数据（按照preprocess_dataset.py的格式）
        labels_per_frame = [labels]  # 每个样本只有一个"帧"
        frame_timestamps_us = np.array([sample.start_time_us], dtype=np.int64)
        
        # 5. 保存标签
        labels_dir = sequence_dir / 'labels_v2'
        labels_dir.mkdir(parents=True, exist_ok=True)
        self.save_labels(labels_dir, labels_per_frame, frame_timestamps_us)
        
        # 6. 生成事件表示时间戳
        ev_repr_timestamps_us = self.generate_ev_repr_timestamps(sample.start_time_us, sample.end_time_us)
        
        # 7. 写入事件表示
        ev_repr_dir = sequence_dir / 'event_representations_v2'
        self.write_event_representations(events, ev_repr_timestamps_us, ev_repr_dir)
        
        # 8. 生成objframe_idx_2_repr_idx映射
        # 确保objframe_idx_2_repr_idx的长度与帧数量匹配
        # 每个样本只有一个帧，所以objframe_idx_2_repr_idx只有一个元素0
        objframe_idx_2_repr_idx = np.zeros(1, dtype=np.int64)
        np.save(str(ev_repr_dir / f'stacked_histogram_dt={TS_STEP_EV_REPR_MS}_nbins={EV_REPR_NBINS}/objframe_idx_2_repr_idx.npy'), 
                objframe_idx_2_repr_idx)
    
    def convert(self, train_ratio: float = 0.7, val_ratio: float = 0.2):
        """
        完整转换流程
        """
        # 1. 加载并降采样数据
        data = self.load_and_concat_h5_files()
        
        # 2. 创建样本
        samples = self.create_samples(data)
        
        # 3. 划分数据集
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            'train': samples[:n_train],
            'val': samples[n_train:n_train + n_val],
            'test': samples[n_train + n_val:]
        }
        
        print(f"\n数据集划分:")
        print(f"  训练: {len(splits['train'])}")
        print(f"  验证: {len(splits['val'])}")
        print(f"  测试: {len(splits['test'])}")
        
        # 4. 处理每个样本
        for split_name, split_samples in splits.items():
            print(f"\n处理{split_name}集...")
            split_out_dir = self.output_dir / split_name
            split_out_dir.mkdir(parents=True, exist_ok=True)
            for sample in tqdm(split_samples, desc=f"  处理{split_name}"):
                self.process_sample(sample, Path(split_out_dir))
        
        # 5. 生成数据集元信息文件
        self.generate_dataset_info(splits)
        
        print(f"\n转换完成！输出目录: {self.output_dir}")
        print(f"\n数据集结构:")
        print(f"  {self.output_dir}/")
        print(f"  ├── train/")
        print(f"  │   ├── sequence_000001/")
        print(f"  │   │   ├── labels_v2/")
        print(f"  │   │   │   ├── labels.npz")
        print(f"  │   │   │   └── timestamps_us.npy")
        print(f"  │   │   └── event_representations_v2/")
        print(f"  │   │       └── stacked_histogram_dt=50_nbins=10/")
        print(f"  │   │           ├── event_representations.h5")
        print(f"  │   │           ├── objframe_idx_2_repr_idx.npy")
        print(f"  │   │           └── timestamps_us.npy")
        print(f"  │   └── sequence_000002/")
        print(f"  │       └── ...")
        print(f"  ├── val/")
        print(f"  │   └── ...")
        print(f"  └── test/")
        print(f"      └── ...")
        print(f"  └── dataset_info.json")
    
    def generate_dataset_info(self, splits: Dict[str, List[DASSample]]):
        """
        生成数据集元信息文件
        """
        import json
        
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
                "ts_step_ev_repr_ms": TS_STEP_EV_REPR_MS,
                "ev_repr_nbins": EV_REPR_NBINS,
                "threshold_percentile": self.threshold_percentile
            },
            "splits": {
                "train": {
                    "count": len(splits["train"]),
                    "sequences": [f"sequence_{s.sample_id:06d}" for s in splits["train"]]
                },
                "val": {
                    "count": len(splits["val"]),
                    "sequences": [f"sequence_{s.sample_id:06d}" for s in splits["val"]]
                },
                "test": {
                    "count": len(splits["test"]),
                    "sequences": [f"sequence_{s.sample_id:06d}" for s in splits["test"]]
                }
            },
            "total_sequences": sum(len(v) for v in splits.values())
        }
        
        # 保存数据集信息
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n数据集信息已保存到: {info_file}")
 
 
# =============================================================
 
def main():
    parser = argparse.ArgumentParser(description='DAS瀑布图转st-yolo格式')
    parser.add_argument('--input_pattern', default='data/H5/*.h5', help='输入H5文件模式')
    parser.add_argument('--output_dir', default='data/st_yolo_das', help='输出目录')
    parser.add_argument('--threshold_percentile', type=float, default=95.0, help='异常检测阈值百分位')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    
    args = parser.parse_args()
    
    converter = DASConverter(
        input_h5_pattern=args.input_pattern,
        output_base_dir=args.output_dir,
        threshold_percentile=args.threshold_percentile
    )
    
    converter.convert(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
 
 
if __name__ == "__main__":
    main()
