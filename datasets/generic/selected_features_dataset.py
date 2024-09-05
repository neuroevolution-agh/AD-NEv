import math
from abc import ABC
from typing import List
import torch
from torch.utils.data import Dataset


class SelectedFeaturesDataset(Dataset, ABC):
    def __init__(self, dataset, selected_features: List[int], transform=None):
        self._dataset = dataset
        self.selected_features = selected_features
        self.transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample, label = self._dataset[idx]
        sample_with_selected_features = sample[:, self.selected_features]
        return sample_with_selected_features, label
