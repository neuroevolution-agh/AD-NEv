import os
from typing import List, Tuple

import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from datasets.generic.overlap_windows_dataset import OverlapWindowsDataset
from datasets.smap_msl.smap_msl_data_creator import all_channels, loaders_dir, create_datasets
from datasets.smap_msl.smap_msl_dataset import SmapMslDataset
import numpy as np


def split_smap_msl_validations(datasets, validation_size=0.2, part_dataset=1):
    train_datasets = []
    validation_datasets = []
    for dataset in datasets:
        end_point = int(len(dataset) * part_dataset)
        split_point = int((1 - validation_size) * end_point)
        train_data, train_labels = dataset[:split_point]
        validation_data, validation_labels = dataset[split_point:end_point]
        train_datasets.append(SmapMslDataset(train_data, train_labels))
        validation_datasets.append(SmapMslDataset(validation_data, validation_labels))
    return train_datasets, validation_datasets


def split_smap_msl_validation(dataset, validation_size=0.2, part_dataset=1):
    end_point = int(len(dataset) * part_dataset)
    split_point = int((1 - validation_size) * end_point)
    train_data, train_labels = dataset[:split_point]
    validation_data, validation_labels = dataset[split_point:end_point]
    return SmapMslDataset(train_data, train_labels), SmapMslDataset(validation_data, validation_labels)


def create_smap_msl_loader(train_dataset, test_dataset, window_size) -> Tuple[DataLoader, DataLoader]:
    train_loaders = DataLoader(OverlapWindowsDataset(train_dataset, window_size), shuffle=False, batch_size=16,
                               drop_last=True)

    if test_dataset is not None:
        test_loaders = DataLoader(OverlapWindowsDataset(test_dataset, window_size), shuffle=False, batch_size=1,
                                  drop_last=True)

    else:
        test_loaders = None
    return train_loaders, test_loaders


def create_loaders(train_datasets, test_datasets, window_size) -> Tuple[List[DataLoader], List[DataLoader]]:
    train_loaders = [DataLoader(OverlapWindowsDataset(t, window_size), shuffle=False, batch_size=16, drop_last=True)
                     for t in train_datasets]

    if test_datasets is not None:
        test_loaders = [DataLoader(OverlapWindowsDataset(t, window_size), shuffle=False, batch_size=1, drop_last=True)
                        for t
                        in test_datasets]
    else:
        test_loaders = None
    return train_loaders, test_loaders


def create_concatenate_smap_msl_loader(train_datasets, test_datasets, window_size) -> Tuple[DataLoader, DataLoader]:
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


def get_smap_datasets(_should_read_data_from_files=False, downsample=False) -> Tuple[List[Dataset], List[Dataset], List]:
    return _get_datasets('SMAP', _should_read_data_from_files, downsample)


def get_msl_datasets(_should_read_data_from_files=False, downsample=False) -> Tuple[List[Dataset], List[Dataset], List]:
    return _get_datasets('MSL', _should_read_data_from_files, downsample)


def get_msl_one_common_datasets(_should_read_data_from_files=False, downsample=False) -> Tuple[
    SmapMslDataset, SmapMslDataset]:
    return _get_one_common_datasets('MSL', _should_read_data_from_files, downsample)


def get_smap_one_common_datasets(_should_read_data_from_files=False, downsample=False) -> Tuple[
    SmapMslDataset, SmapMslDataset]:
    return _get_one_common_datasets('SMAP', _should_read_data_from_files, downsample)


def _get_datasets(spacecraft, _should_read_data_from_files=False, downsample=False) -> Tuple[
    List[Dataset], List[Dataset], List]:
    if _should_read_data_from_files:
        create_datasets(downsample)
    channels = all_channels(spacecraft)
    loaders_dir_path = loaders_dir(spacecraft)
    train_datasets = []
    test_datasets = []
    channels_name = []
    for channel in channels:
        # if channel == 'P-1':
        train_dataset = torch.load(os.path.join(loaders_dir_path, f'{channel}-data-train.pt'))
        test_dataset = torch.load(os.path.join(loaders_dir_path, f'{channel}-data-test.pt'))
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        channels_name.append(channel)

    return train_datasets, test_datasets, channels_name


def _get_one_common_datasets(spacecraft, _should_read_data_from_files=False, downsample=False) -> Tuple[
    SmapMslDataset, SmapMslDataset]:
    if _should_read_data_from_files:
        create_datasets(downsample)
    channels = all_channels(spacecraft)
    loaders_dir_path = loaders_dir(spacecraft)
    train_datasets = []
    test_datasets = []
    label_train_data = []
    label_test_data = []
    for channel in channels:
        train_dataset = torch.load(os.path.join(loaders_dir_path, f'{channel}-data-train.pt'))
        test_dataset = torch.load(os.path.join(loaders_dir_path, f'{channel}-data-test.pt'))
        train_datasets.extend(train_dataset.data_from_sensor.numpy())
        test_datasets.extend(test_dataset.data_from_sensor.numpy())
        label_train_data.extend(train_dataset.labels.numpy())
        label_test_data.extend(test_dataset.labels.numpy())

    train_datasets = np.asarray(train_datasets)
    test_datasets = np.asarray(test_datasets)
    # train_datasets, test_datasets = _normalize_data_per_sensor_train_test(train_datasets, test_datasets)
    return SmapMslDataset(train_datasets, label_train_data), SmapMslDataset(test_datasets, label_test_data)


def _normalize_data_per_sensor_train_test(train_input_as_matrix, test_input_as_matrix):
    result_train = np.empty([train_input_as_matrix.shape[0], train_input_as_matrix.shape[1]])
    result_test = np.empty([test_input_as_matrix.shape[0], train_input_as_matrix.shape[1]])

    for index in range(train_input_as_matrix.shape[1]):
        sensor_values_train = train_input_as_matrix[:, index]
        sensor_values_test = test_input_as_matrix[:, index]
        sensor_scaler = MinMaxScaler(feature_range=(-1, 1))
        sensor_scaler.fit(sensor_values_train.reshape(-1, 1))
        sensor_scaler.partial_fit(sensor_values_test.reshape(-1, 1))

        normalized_train = sensor_scaler.transform(sensor_values_train.reshape(-1, 1))
        result_train[:, index] = normalized_train.reshape(-1)

        normalized_test = sensor_scaler.transform(sensor_values_test.reshape(-1, 1))
        result_test[:, index] = normalized_test.reshape(-1)
    return result_train, result_test
