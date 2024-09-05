from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from statistics.downsample import median_downsample

swat_sensor_names = ['FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201',
                'P202', 'P203', 'P204', 'P205', 'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303',
                'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404',
                'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502',
                'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603']

swat_senosr_names_with_atttack = ['MV101', 'P102', 'LIT101', 'AIT202', 'LIT301', 'DPIT301', 'FIT401', 'MV304', 'MV303',
                                  'AIT504', 'UV401', 'AIT502', 'PIT501', 'P602', 'P203', 'P205', 'P401', 'LIT401',
                                  'P302', 'P501', 'FIT502', 'AIT402']

label_name = 'Normal/Attack'

swat_training_data_path = os.path.join(os.getcwd(), 'resources', 'data', 'SWAT', 'normal')
# data_path = os.path.join(os.path.dirname(__file__), 'resources', 'data_temp')
swat_training_filenames = glob.glob(swat_training_data_path + "/SWaT_Dataset_Normal_v1.xlsx")

swat_test_data_path = os.path.join(os.getcwd(), 'resources', 'data', 'SWAT', 'attack')
# data_path = os.path.join(os.path.dirname(__file__), 'resources', 'data_temp')
swat_test_filenames = glob.glob(swat_test_data_path + "/*.xlsx")

scaler = MinMaxScaler(feature_range=(-1, 1))
number_of_swat_sensor = len(swat_sensor_names)


def prepare_swat_train_test(train_data_from_file, test_data_from_file, downsample=False):
    train_data, train_labels = prepare_swat_data(train_data_from_file, downsample=downsample)
    test_data, test_labels = prepare_swat_data(test_data_from_file)

    train_data, test_data = _normalize_data_per_sensor_train_test(train_data, test_data)

    return train_data, train_labels, test_data, test_labels


def prepare_swat_data(data_from_file, downsample=False):
    number_of_survey = len(data_from_file)
    row_data = np.empty(number_of_swat_sensor)
    input_as_matrix = np.empty([number_of_survey, number_of_swat_sensor])
    labels = np.empty(number_of_survey)
    for row in range(len(data_from_file)):
        for index, sensor in enumerate(swat_sensor_names):
            row_data[index] = data_from_file[row][sensor]
        input_as_matrix[row, :] = row_data
        labels[row] = 0 if data_from_file[row][label_name].lower() == 'Normal'.lower() else 1
    if downsample:
        input_as_matrix = median_downsample(input_as_matrix, 5)
    # normalized_input = _normalizeDataPerSensor(input_as_matrix, input_as_matrix.shape[0])
    return input_as_matrix, labels


def _normalizeDataPerSensor(input_as_matrix, number_of_survey):
    result = np.empty([number_of_survey, number_of_swat_sensor])
    for index in range(number_of_swat_sensor):
        sensor_values = input_as_matrix[:, index]
        normalized_input = scaler.fit_transform(sensor_values.reshape(-1, 1))
        result[:, index] = normalized_input.reshape(-1)
    return result


def _normalize_data_per_sensor_train_test(train_input_as_matrix, test_input_as_matrix):
    result_train = np.empty([train_input_as_matrix.shape[0], number_of_swat_sensor])
    result_test = np.empty([test_input_as_matrix.shape[0], number_of_swat_sensor])

    for index in range(number_of_swat_sensor):
        sensor_values_train = train_input_as_matrix[:, index]
        sensor_values_test = test_input_as_matrix[:, index]
        sensor_scaler = MinMaxScaler(feature_range=(0, 1))
        sensor_scaler.fit(sensor_values_train.reshape(-1, 1))
        sensor_scaler.partial_fit(sensor_values_test.reshape(-1, 1))

        normalized_train = sensor_scaler.transform(sensor_values_train.reshape(-1, 1))
        result_train[:, index] = normalized_train.reshape(-1)

        normalized_test = sensor_scaler.transform(sensor_values_test.reshape(-1, 1))
        result_test[:, index] = normalized_test.reshape(-1)
    return result_train, result_test


def prepare_swat_raw_data(data_from_file):
    number_of_survey = len(data_from_file)
    row_data = np.empty(number_of_swat_sensor)
    input_as_matrix = np.empty([number_of_survey, number_of_swat_sensor])
    labels = np.empty(number_of_survey)
    for row in range(len(data_from_file)):
        for index, sensor in enumerate(swat_sensor_names):
            row_data[index] = data_from_file[row][sensor]
        input_as_matrix[row, :] = row_data
        labels[row] = 0 if data_from_file[row][label_name].lower() == 'Normal'.lower() else 1
    return input_as_matrix, labels


if __name__ == '__main__':
    indexes = []
    indexes_with_name = dict()
    for sensor in swat_senosr_names_with_atttack:
        indices = [i for i, x in enumerate(swat_sensor_names) if x == sensor]
        indexes.append(indices.__getitem__(0))
        indexes_with_name.setdefault(sensor, indices.__getitem__(0))

    indexes.sort()
    print(indexes_with_name)