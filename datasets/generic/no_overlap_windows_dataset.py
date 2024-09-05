import math
from abc import ABC

import torch
from torch.utils.data import Dataset


class NoOverlapWindowsDataset(Dataset, ABC):
    def __init__(self, dataset, window_size=1, transform=None):
        self._dataset = dataset
        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        return math.floor(len(self._dataset) / self.window_size)

    def __getitem__(self, idx):
        window_start = idx * self.window_size
        window_end = (idx + 1) * self.window_size
        data_window, labels_window = self._dataset[window_start: window_end]
        label = 1 if torch.any(labels_window == 1) else 0
        return data_window, label

