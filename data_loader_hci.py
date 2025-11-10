import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 定义数据集路径和文件名模式
file_prefix = "Part_"
file_extension = "_processed.npy"

excluded_indices = {12, 15, 26}

def load_hci_data(file_path):
    """加载HCI的.npy数据文件"""
    try:
        data_dict = np.load(file_path, allow_pickle=True).item()
        trails = data_dict['trails']  # 形状为 [trials, channels, samples]
        labels = data_dict['labels']  # 形状为 [trials]
        # 标签归一化：小于5的为0，大于等于5的为1
        labels = (labels >= 5).astype(np.int64)
        return trails, labels
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None


def preprocess_data(data, labels, time_window, sampling_rate=256, ):
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
    all_data = []
    all_labels = []
    for num_trail in range(data.shape[0]):
        length_windows = sampling_rate * time_window
        num_windows = data[num_trail].shape[1]//length_windows
        # Reshape data into windows
        datacut = data[num_trail][:, :num_windows * length_windows]
        datacut = datacut.reshape(datacut.shape[0], num_windows, length_windows)
        datacut = datacut.transpose(1, 0, 2)

        # Expand labels to match the number of windows
        labelscut = np.repeat(labels[num_trail, :][np.newaxis, :], num_windows, axis=0)

        all_data.append(datacut)
        all_labels.append(labelscut)

    all_data = np.concatenate(all_data, axis=0)  # [total_windows, channels, window_samples]
    all_labels = np.concatenate(all_labels, axis=0)

    return all_data, all_labels


class HCIDataset(Dataset):
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


def get_dataloaders_loso_hci(file_range, current_idx, batch_size=32, time_window=4, label_index=1, data_dir=None, num_workers=0):
    # 获取训练集和测试集文件
    train_files = [f"{file_prefix}{i}{file_extension}" for i in range(1, file_range+1) if i != current_idx and i not in excluded_indices]
    test_file = f"{file_prefix}{current_idx}{file_extension}"

    # 加载训练集数据
    train_data = []
    train_labels = []
    for file_name in train_files:
        file_path = os.path.join(data_dir, file_name)
        data, labels = load_hci_data(file_path)

        data, labels = preprocess_data(data, labels, time_window)

        if data is not None and labels is not None:
            train_data.append(data)
            train_labels.append(labels)

    # 加载测试集数据
    test_file_path = os.path.join(data_dir, test_file)
    test_data, test_labels = load_hci_data(test_file_path)
    test_data, test_labels = preprocess_data(test_data, test_labels, time_window)

    # 检查数据是否为空
    if not train_data or test_data is None or test_labels is None:
        raise ValueError("Missing data for training or testing set!")

    # 合并训练集数据
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # 创建数据集和数据加载器
    train_dataset = HCIDataset(train_data, train_labels, label_index)
    test_dataset = HCIDataset(test_data, test_labels, label_index)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)

    return train_dataloader, test_dataloader
