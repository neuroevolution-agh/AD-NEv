from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path
import glob
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from statistics.downsample import median_downsample

Path()

wadi_training_data_path = os.path.join(os.getcwd(), 'resources', 'data', 'WADI', 'normal')
# data_path = os.path.join(os.path.dirname(__file__), 'resources', 'data_temp')
wadi_training_filenames = glob.glob(wadi_training_data_path + "/*.xlsx")

wadi_test_data_path = os.path.join(os.getcwd(), 'resources', 'data', 'WADI', 'attack')
# data_path = os.path.join(os.path.dirname(__file__), 'resources', 'data_temp')
wadi_test_filenames = glob.glob(wadi_test_data_path + "/*.xlsx")

wadi_senosr_names = ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV', '1_FIT_001_PV',
                     '1_LS_001_AL', '1_LS_002_AL', '1_LT_001_PV', '1_MV_001_STATUS', '1_MV_002_STATUS',
                     '1_MV_003_STATUS',
                     '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_002_STATUS', '1_P_003_STATUS', '1_P_004_STATUS',
                     '1_P_005_STATUS', '1_P_006_STATUS', '2_DPIT_001_PV', '2_FIC_101_CO', '2_FIC_101_PV',
                     '2_FIC_101_SP', '2_FIC_201_CO', '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV',
                     '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', '2_FIC_501_CO', '2_FIC_501_PV',
                     '2_FIC_501_SP', '2_FIC_601_CO', '2_FIC_601_PV', '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV',
                     '2_FIT_003_PV', '2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', '2_FQ_501_PV',
                     '2_FQ_601_PV', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH',
                     '2_LS_201_AL', '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH',
                     '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_LT_001_PV', '2_LT_002_PV', '2_MCV_007_CO',
                     '2_MCV_101_CO', '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO',
                     '2_MV_001_STATUS', '2_MV_002_STATUS', '2_MV_003_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS',
                     '2_MV_006_STATUS', '2_MV_009_STATUS', '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS',
                     '2_MV_401_STATUS', '2_MV_501_STATUS', '2_MV_601_STATUS',
                     '2_P_003_SPEED', '2_P_003_STATUS', '2_P_004_SPEED', '2_P_004_STATUS', '2_PIC_003_CO',
                     '2_PIC_003_PV', '2_PIC_003_SP', '2_PIT_001_PV', '2_PIT_002_PV', '2_PIT_003_PV', '2_SV_101_STATUS',
                     '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS', '2_SV_601_STATUS',
                     '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV', '2B_AIT_001_PV',
                     '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV', '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV',
                     '3_AIT_004_PV', '3_AIT_005_PV', '3_FIT_001_PV', '3_LS_001_AL', '3_LT_001_PV', '3_MV_001_STATUS',
                     '3_MV_002_STATUS', '3_MV_003_STATUS', '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS',
                     '3_P_004_STATUS']

wadi_senosr_names_with_atttack = ['1_LT_001_PV', '1_FIT_001_PV', '2_LT_002_PV', '1_AIT_001_PV', '1_AIT_002_PV',
                                  '2_MCV_101_CO',
                                  '2_MCV_201_CO', '2_MCV_301_CO',
                                  '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO', '2_MV_003_STATUS', '2_PIT_002_PV',
                                  '2_FIT_002_PV',
                                  '1_P_006_STATUS', '1_MV_001_STATUS', '2_FIT_003_PV', '2_PIT_003_PV']

label_name = 'Label'
number_of_wadi_sensor = len(wadi_senosr_names)
scaler = MinMaxScaler(feature_range=(-1, 1))


def prepare_wadi_train_test(train_data_from_file, test_data_from_file, downsample=False):
    train_data, train_labels = prepare_wadi_data(train_data_from_file, downsample=downsample, isTraining=True)
    test_data, test_labels = prepare_wadi_data(test_data_from_file, isTraining=False)

    train_data, test_data = _normalize_data_per_sensor_train_test(train_data, test_data)

    return train_data, train_labels, test_data, test_labels


def prepare_wadi_data(data_from_file, downsample=False, isTraining=True):
    number_of_survey = len(data_from_file)
    row_data = np.empty(number_of_wadi_sensor)
    input_as_matrix = np.empty([number_of_survey, number_of_wadi_sensor])
    labels = np.empty(number_of_survey)
    for row in range(len(data_from_file)):
        for index, sensor in enumerate(wadi_senosr_names):
            try:
                row_data[index] = float(data_from_file[row][sensor])
            except ValueError:
                row_data[index] = float(data_from_file[row - 1][sensor])
                data_from_file[row][sensor] = float(data_from_file[row - 1][sensor])
        input_as_matrix[row, :] = row_data
        if isTraining:
            labels[row] = 0
        else:
            labels[row] = 0 if data_from_file[row][label_name] == '1' else 1
    if downsample:
        input_as_matrix = median_downsample(input_as_matrix, 5)
    # normalized_input = _normalizeDataPerSensor(input_as_matrix, input_as_matrix.shape[0])
    return input_as_matrix, labels


def _normalize_data_per_sensor_train_test(train_input_as_matrix, test_input_as_matrix):
    result_train = np.empty([train_input_as_matrix.shape[0], number_of_wadi_sensor])
    result_test = np.empty([test_input_as_matrix.shape[0], number_of_wadi_sensor])

    for index in range(number_of_wadi_sensor):
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


def _normalizeDataPerSensor(input_as_matrix, number_of_survey):
    result = np.empty([number_of_survey, number_of_wadi_sensor])
    for index in range(number_of_wadi_sensor):
        sensor_values = input_as_matrix[:, index]
        normalized_input = scaler.fit_transform(sensor_values.reshape(-1, 1))
        result[:, index] = normalized_input.reshape(-1)
    return result


if __name__ == '__main__':
    indexes = []
    indexes_with_name = dict()
    for sensor in wadi_senosr_names_with_atttack:
        indices = [i for i, x in enumerate(wadi_senosr_names) if x == sensor]
        indexes.append(indices.__getitem__(0))
        indexes_with_name.setdefault(sensor, indices.__getitem__(0))

    indexes.sort()
    print(indexes_with_name)

    wadi_groups = [[8, 12, 20, 21, 29, 30, 41, 44, 52, 64, 67, 69, 75, 78, 87, 111, 112, 115, 119],
                   [3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 24, 27, 33, 36, 38, 40, 42, 43, 45, 46, 47, 49,
                    60, 61, 70, 72, 73, 81, 82, 83, 85, 86, 88, 89, 90, 96, 98, 118],
                   [48, 50, 53, 54, 58, 68, 71, 72, 84, 94, 102, 105, 107, 108, 113, 116, 117],
                   [1, 22, 23, 25, 26, 28, 31, 32, 34, 35, 37, 39, 59, 62, 63, 65, 66, 76, 77, 79, 80, 92, 93, 95, 106,
                    109, 110], [0, 2, 7, 51, 57, 74, 91, 97, 99, 100, 101, 103, 104, 114]]
#
#
#     1 - 2
#     2 - 8
#     3 - 0
#     4 - 6
#     5 - 1



