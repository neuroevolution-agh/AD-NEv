import json
import os
import string
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from datasets.smap_msl.smap_msl_dataset import SmapMslDataset
from datasets.smd.smd_msl_dataset import SmdDataset

from statistics.downsample import median_downsample

number_of_smd_sensor = 38


def create_datasets(downsample=False):
    training_data_path = os.path.join(os.getcwd(), 'resources', 'data', 'smd', 'train')
    test_data_path = os.path.join(os.getcwd(), 'resources', 'data', 'smd', 'test')
    test_label_path = os.path.join(os.getcwd(), 'resources', 'data', 'smd', 'test_label')
    machines = os.listdir(test_label_path)
    for machine in machines:
        train_dataset, test_dataset = _load_data_for_machine(training_data_path, test_data_path, test_label_path,
                                                             machine, machines, downsample)

        loaders_dir_path = loaders_dir()
        if not os.path.exists(loaders_dir_path): os.mkdir(loaders_dir_path)
        machine = machine.replace(".txt", "")

        torch.save(train_dataset, os.path.join(loaders_dir_path, f'{machine}-data-train.pt'))
        torch.save(test_dataset, os.path.join(loaders_dir_path, f'{machine}-data-test.pt'))


def _load_machines():
    data_dir_path = os.path.join(os.getcwd(), 'resources', 'data', 'smd', 'test_label')
    return os.listdir(data_dir_path)


def _load_data_for_machine(train_dir, test_dir, label_dir, channel: str, anomalies, downsample=False):
    data_train = _load_data(train_dir, channel)
    if downsample:
        data_train = median_downsample(data_train, 5)
    data_test = _load_data(test_dir, channel)
    labels_test = _load_data(label_dir, channel)
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # data_train = scaler.fit_transform(data_train)
    # data_test = scaler.fit_transform(data_test)
    #cdata_train, data_test = _normalize_data_per_sensor_train_test(data_train, data_test)
    labels_train = [0] * len(data_train)
    assert len(data_train) == len(labels_train)
    assert len(data_test) == len(labels_test)
    train_dataset = SmdDataset(data_train, labels_train)
    test_dataset = SmdDataset(data_test, labels_test)
    return train_dataset, test_dataset


def _load_data(dir, channel):
    file = os.path.join(dir, f'{channel}')
    return np.loadtxt(file, delimiter=',')


def all_machines():
    anomalies = _load_machines()
    return anomalies


def loaders_dir():
    return os.path.join('resources', 'dataloader', 'SMD')


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
