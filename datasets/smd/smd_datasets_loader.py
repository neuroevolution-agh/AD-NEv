import os
import string
from typing import List, Tuple

import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from datasets.generic.overlap_windows_dataset import OverlapWindowsDataset
import numpy as np

from datasets.smd.smd_data_creator import create_datasets, all_machines
from datasets.smd.smd_msl_dataset import SmdDataset


def split_smd_validations(datasets, validation_size=0.2, part_dataset=1):
    train_datasets = []
    validation_datasets = []
    for dataset in datasets:
        end_point = int(len(dataset) * part_dataset)
        split_point = int((1 - validation_size) * end_point)
        train_data, train_labels = dataset[:split_point]
        validation_data, validation_labels = dataset[split_point:end_point]
        train_datasets.append(SmdDataset(train_data, train_labels))
        validation_datasets.append(SmdDataset(validation_data, validation_labels))
    return train_datasets, validation_datasets


def split_smd_validation(dataset, validation_size=0.2, part_dataset=1):
    end_point = int(len(dataset) * part_dataset)
    split_point = int((1 - validation_size) * end_point)
    train_data, train_labels = dataset[:split_point]
    validation_data, validation_labels = dataset[split_point:end_point]
    return SmdDataset(train_data, train_labels), SmdDataset(validation_data, validation_labels)


def create_smd_loaders(train_datasets, test_datasets, window_size) -> Tuple[List[DataLoader], List[DataLoader]]:
    train_loaders = [DataLoader(OverlapWindowsDataset(t, window_size), shuffle=False, batch_size=16, drop_last=True)
                     for t in train_datasets]

    if test_datasets is not None:
        test_loaders = [DataLoader(OverlapWindowsDataset(t, window_size), shuffle=False, batch_size=1, drop_last=True)
                        for t
                        in test_datasets]
    else:
        test_loaders = None
    return train_loaders, test_loaders


def create_smd_loader(train_datasets, test_datasets, window_size) -> Tuple[DataLoader, DataLoader]:
    OverlapTrainDatasets = [(OverlapWindowsDataset(t, window_size))
                            for t in train_datasets]

    concatenated_train_ds = torch.utils.data.ConcatDataset(OverlapTrainDatasets)

    train_loader = DataLoader(concatenated_train_ds, shuffle=False, batch_size=16, drop_last=True)
    if test_datasets is not None:
        OverlapTestDatasets = [(OverlapWindowsDataset(t, window_size)) for t
                               in test_datasets]
        concatenated_test_ds = torch.utils.data.ConcatDataset(OverlapTestDatasets)
        test_loader = DataLoader(concatenated_test_ds, shuffle=False, batch_size=1, drop_last=True)
    else:
        test_loader = None

    return train_loader, test_loader


def get_smd_datasets(_should_read_data_from_files=False, downsample=False) -> Tuple[List[Dataset], List[Dataset], List]:
    return _get_datasets(_should_read_data_from_files, downsample)


def _get_datasets(_should_read_data_from_files=False, downsample=False) -> Tuple[
    List[Dataset], List[Dataset], List]:
    if _should_read_data_from_files:
        create_datasets(downsample)
    machines = all_machines()
    loaders_dir_path = os.path.join('resources', 'dataloader', 'SMD')
    train_datasets = []
    test_datasets = []
    machines_name = []
    for machine in machines:
        machine = machine.replace(".txt", "")
        train_dataset = torch.load(os.path.join(loaders_dir_path, f'{machine}-data-train.pt'))
        test_dataset = torch.load(os.path.join(loaders_dir_path, f'{machine}-data-test.pt'))
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        machines_name.append(machine)


    return train_datasets, test_datasets, machines_name


def _get_one_common_datasets(_should_read_data_from_files=False, downsample=False) -> Tuple[
    SmdDataset, SmdDataset]:
    if _should_read_data_from_files:
        create_datasets(downsample)
    channels = all_machines()
    loaders_dir_path = os.path.join('resources', 'dataloader', 'SMD')
    train_datasets = []
    test_datasets = []
    label_train_data = []
    label_test_data = []
    for channel in channels:
        train_dataset = torch.load(os.path.join(loaders_dir_path, f'{channel}-data-train.pt'))
        test_dataset = torch.load(os.path.join(loaders_dir_path, f'{channel}-data-test.pt'))
        train_datasets.extend(train_dataset.data.numpy())
        test_datasets.extend(test_dataset.data.numpy())
        label_train_data.extend(train_dataset.labels.numpy())
        label_test_data.extend(test_dataset.labels.numpy())

    train_datasets = np.asarray(train_datasets)
    test_datasets = np.asarray(test_datasets)
    return SmdDataset(train_datasets, label_train_data), SmdDataset(test_datasets, label_test_data)


