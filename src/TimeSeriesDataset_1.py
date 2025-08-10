import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, device):
        """
        data: 时间序列数据，假设为 NumPy 数组。
        input_window: 使用的时间窗口大小（i）。
        """
        self.data = data
        self.input_window = input_window

    def __len__(self):
        # 确保有足够的数据来预测
        return len(self.data) - self.input_window

    def __getitem__(self, idx):
        # 获取输入（特征）和输出（标签）
        x = self.data[idx:idx+self.input_window]
        y = self.data[idx+self.input_window]
        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)

# # 创建 DataLoader
# def create_dataloader(data, input_window, batch_size=32):
#     dataset = TimeSeriesDataset(data, input_window)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 示例：创建 DataLoader
# data = np.load("your_timeseries_data.npy")  # 加载您的数据
# dataloader = create_dataloader(data, input_window=您的i值, batch_size=32)
