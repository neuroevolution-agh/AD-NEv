from abc import ABC

import torch
from torch.utils.data import Dataset


class WadiDataset(Dataset, ABC):
    def __init__(self, data_from_sensor, labels, transform=None):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.data_from_sensor = torch.tensor(data_from_sensor, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return self.data_from_sensor.size(0)

    def __getitem__(self, idx):
        return self.data_from_sensor[idx], self.labels[idx]
