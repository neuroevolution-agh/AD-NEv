import json
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from datasets.smap_msl.smap_msl_dataset import SmapMslDataset

from statistics.downsample import median_downsample

number_of_msl_sensor = 55
number_of_smap_sensor = 25

def create_datasets(downsample=False):
    training_data_path = os.path.join(os.getcwd(), 'resources', 'data', 'smap_msl', 'train')
    test_data_path = os.path.join(os.getcwd(), 'resources', 'data', 'smap_msl', 'test')
    anomalies = _load_anomalies()
    channels = anomalies.index.unique()
    for channel in channels:
        train_dataset, test_dataset = _load_data_for_channel(training_data_path, test_data_path, channel, anomalies, downsample)

        loaders_dir_path = loaders_dir(anomalies.loc[channel]['spacecraft'])
        if not os.path.exists(loaders_dir_path): os.mkdir(loaders_dir_path)

        torch.save(train_dataset, os.path.join(loaders_dir_path, f'{channel}-data-train.pt'))
        torch.save(test_dataset, os.path.join(loaders_dir_path, f'{channel}-data-test.pt'))


def _load_anomalies():
    data_dir_path = os.path.join(os.getcwd(), 'resources', 'data', 'smap_msl')
    anomalies_path = os.path.join(data_dir_path, 'labeled_anomalies.csv')
    anomalies = pd.read_csv(anomalies_path, index_col='chan_id')
    anomalies['anomaly_sequences'] = anomalies['anomaly_sequences'].apply(lambda t: json.loads(t))
    return anomalies


def _load_data_for_channel(train_dir, test_dir, channel: str, anomalies: pd.DataFrame, downsample=False):
    data_train = _load_data(train_dir, channel)
    if downsample:
        data_train = median_downsample(data_train, 5)
    data_test = _load_data(test_dir, channel)
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # data_train = scaler.fit_transform(data_train)
    # data_test = scaler.fit_transform(data_test)
    data_train, data_test = _normalize_data_per_sensor_train_test(data_train, data_test)
    labels_train = [0] * len(data_train)
    labels_test = _create_test_labels(anomalies, channel)
    assert len(data_train) == len(labels_train)
    assert len(data_test) == len(labels_test)
    train_dataset = SmapMslDataset(data_train, labels_train)
    test_dataset = SmapMslDataset(data_test, labels_test)
    return train_dataset, test_dataset


def _create_test_labels(anomalies: pd.DataFrame, channel: str) -> List[int]:
    sequences: List[List[int]] = anomalies.loc[channel]['anomaly_sequences']
    limit = anomalies.loc[channel]['num_values']
    labels = []
    for seq in sequences:
        current_length = len(labels)
        start = seq[0]
        end = seq[1]
        labels.extend((start - current_length) * [0])
        labels.extend((end - start) * [1])
    labels.extend((limit - len(labels)) * [0])
    return labels


def _load_data(dir, channel):
    file = os.path.join(dir, f'{channel}.npy')
    return np.load(file)


def all_channels(spacecraft):
    anomalies = _load_anomalies()
    return anomalies[anomalies['spacecraft'] == spacecraft].index.unique()


def loaders_dir(spacecraft):
    return os.path.join('resources', 'dataloader', spacecraft)

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


if __name__ == '__main__':
    create_datasets()
