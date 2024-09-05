from typing import List

import torch
from torch.utils.data import DataLoader

from datasets.generic.overlap_windows_dataset import OverlapWindowsDataset
from datasets.generic.selected_features_dataset import SelectedFeaturesDataset


def prepare_loader_with_selected_features(ds, window_size=1, selected_features: List[int] = list, batch_size=16):
    dataset = OverlapWindowsDataset(SelectedFeaturesDataset(ds, selected_features=selected_features),
                                    window_size=window_size)
    return DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True)


def prepare_concatenate_smap_msl_loader_with_selected_features(ds, window_size=1, selected_features: List[int] = list,
                                                               batch_size=16):
    dataset = [(OverlapWindowsDataset(SelectedFeaturesDataset(t, selected_features=selected_features),
                                      window_size=window_size)) for t in ds]

    concatenated_test_ds = torch.utils.data.ConcatDataset(dataset)
    return DataLoader(concatenated_test_ds, shuffle=False, batch_size=batch_size, drop_last=True)


def prepare_loaders_with_selected_features(train_ds=None, test_ds=None, window_size=1,
                                           batch_size=16,
                                           selected_features: List[int] = list):
    train_loader = test_loader = None

    if train_ds is not None:
        train_loader = prepare_loader_with_selected_features(train_ds, window_size=window_size,
                                                             selected_features=selected_features,
                                                             batch_size=batch_size)

    if test_ds is not None:
        test_loader = prepare_loader_with_selected_features(test_ds, window_size=window_size,
                                                            selected_features=selected_features, batch_size=1)

    return train_loader, test_loader


def prepare_loaders_with_selected_features_for_list_ds(train_ds=None, test_ds=None, window_size=1,
                                                       batch_size=16,
                                                       selected_features: List[int] = list):
    train_loader = test_loader = None

    if train_ds is not None:
        train_loader = [prepare_loader_with_selected_features(t, window_size=window_size,
                                                              selected_features=selected_features,
                                                              batch_size=batch_size)
                        for t in train_ds]

    if test_ds is not None:
        test_loader = [prepare_loader_with_selected_features(t, window_size=window_size,
                                                             selected_features=selected_features,
                                                             batch_size=batch_size)
                       for t in test_ds]

    return train_loader, test_loader
