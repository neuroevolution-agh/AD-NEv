import math
from abc import ABC

import torch
from torch.utils.data import Dataset


class OverlapWindowsDataset(Dataset, ABC):
    def __init__(self, dataset, window_size=1, transform=None):
        self._dataset = dataset
        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        return math.floor(len(self._dataset) - self.window_size + 1)

    def __getitem__(self, idx):
        data_window, labels_window = self._dataset[idx:idx + self.window_size]
        label = 1 if labels_window[-1] == 1 else 0
        return data_window, label

    def getitem(self, idx):
        data_window, labels_window = self._dataset[idx:idx + self.window_size]
        return data_window, labels_window
