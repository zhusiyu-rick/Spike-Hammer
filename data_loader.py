import os
import numpy as np
import _pickle as cPickle
from torch.utils.data import Dataset, DataLoader
import torch

# 定义数据目录和文件名前缀

file_prefix = "s"
file_extension = ".dat"


# 定义读取文件的函数
def load_data_from_file(file_path):
    try:
        # 使用 cPickle 读取 .dat 文件
        with open(file_path, 'rb') as f:
            file_content = cPickle.load(f, encoding='latin1')  # DEAP 数据集需要指定 encoding='latin1'

            # 提取 data 和 labels，并转为 NumPy 数组
            data = np.array(file_content.get('data', None))
            labels = np.array(file_content.get('labels', None))

            if data is None or labels is None:
                print(f"Warning: Missing 'data' or 'labels' in {file_path}")

            # 标签归一化：小于5的为0，大于等于5的为1
            labels = (labels >= 5).astype(np.int64)

            return data, labels

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None
    except cPickle.UnpicklingError:
        print(f"Error unpickling file: {file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None, None


def preprocess_data(data, labels, time_window, sampling_rate=128, trim_start_seconds=3):
    """
    Preprocess data by removing the first `trim_start_seconds` seconds and slicing into time windows.

    Args:
        data (np.ndarray): Input data of shape [trials, channels, samples].
        labels (np.ndarray): Corresponding labels of shape [trials].
        time_window (int): Size of the time window (in samples).
        sampling_rate (int): Sampling rate of the signal (default: 128 Hz).
        trim_start_seconds (int): Number of seconds to trim from the start of each trial.

    Returns:
        np.ndarray, np.ndarray: Preprocessed data and expanded labels.
    """
    # Remove the first `trim_start_seconds` seconds
    trim_samples = trim_start_seconds * sampling_rate
    data = data[:, :, trim_samples:]  # Shape: [trials, channels, samples - trim_samples]

    length_windows = sampling_rate * time_window
    num_windows = data.shape[2]//length_windows
    # Reshape data into windows
    data = data.reshape(data.shape[0], data.shape[1], num_windows, length_windows)

    data = data.transpose(0, 2, 1, 3)

    data = data.reshape(-1, data.shape[2], data.shape[3])

    # Expand labels to match the number of windows

    labels = np.repeat(labels, num_windows, axis=0)

    return data, labels


# 自定义数据集类
class DEAPDataset(Dataset):
    def __init__(self, data, labels, label_index=None):
        self.data = torch.tensor(data, dtype=torch.float64)
        self.labels = torch.tensor(labels, dtype=torch.long)
        if label_index is not None:
            if not (0 <= label_index < self.labels.shape[1]):
                raise ValueError(
                    f"Invalid label_index {label_index}. Must be in range [0, {self.labels.shape[1] - 1}].")
            self.labels = self.labels[:, label_index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 用于交叉验证的数据分割函数
def get_dataloaders_loso(file_range, current_idx, batch_size=32, time_window=4, label_index=1, data_dir=None, num_workers=8):
    # 获取训练集和测试集文件
    train_files = [f"{file_prefix}{i:02d}{file_extension}" for i in range(1, file_range+1) if i != current_idx]
    test_file = f"{file_prefix}{current_idx:02d}{file_extension}"

    # 加载训练集数据
    train_data = []
    train_labels = []
    for file_name in train_files:
        file_path = os.path.join(data_dir, file_name)
        data, labels = load_data_from_file(file_path)
        data, labels = preprocess_data(data, labels, time_window)

        if data is not None and labels is not None:
            train_data.append(data)
            train_labels.append(labels)

    # 加载测试集数据
    test_file_path = os.path.join(data_dir, test_file)
    test_data, test_labels = load_data_from_file(test_file_path)
    test_data, test_labels = preprocess_data(test_data, test_labels, time_window)

    # 检查数据是否为空
    if not train_data or test_data is None or test_labels is None:
        raise ValueError("Missing data for training or testing set!")

    # 合并训练集数据
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # 创建数据集和数据加载器
    train_dataset = DEAPDataset(train_data, train_labels, label_index)
    test_dataset = DEAPDataset(test_data, test_labels, label_index)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)

    return train_dataloader, test_dataloader


# 用于交叉验证的数据分割函数
def get_dataloaders_loto(current_trail_idx, current_idx, batch_size=32, time_window=4, label_index=1, data_dir=None, num_workers=8):
    # 获取训练集和测试集文件
    file = f"{file_prefix}{current_idx:02d}{file_extension}"
    file_path = os.path.join(data_dir, file)
    data, labels = load_data_from_file(file_path)
    # data, labels = preprocess_data(data, labels, time_window)
    test_data = data[current_trail_idx:current_trail_idx + 1, :, :]
    test_labels = labels[current_trail_idx:current_trail_idx + 1, :]
    train_data = np.delete(data, current_trail_idx, axis=0)
    train_labels = np.delete(labels, current_trail_idx, axis=0)

    train_data, train_labels = preprocess_data(train_data, train_labels, time_window)
    test_data, test_labels = preprocess_data(test_data, test_labels, time_window)

    # print(train_data.shape, train_labels.shape)
    # print(test_data.shape, test_labels.shape)

    # 创建数据集和数据加载器
    train_dataset = DEAPDataset(train_data, train_labels, label_index)
    test_dataset = DEAPDataset(test_data, test_labels, label_index)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)

    return train_dataloader, test_dataloader


# 示例：获取交叉验证数据加载器

# train_loader, test_loader = get_dataloaders_loso(file_range=32, current_idx=1, batch_size=32, time_window=4, label_index=1, data_dir)
# train_loader, test_loader = get_dataloaders_loto(current_trail_idx=0, current_idx=1, batch_size=32, time_window=4, label_index=1, data_dir)
# for batch_data, batch_labels in train_loader:
#     print(batch_data.shape)
#     print(batch_labels.shape)

