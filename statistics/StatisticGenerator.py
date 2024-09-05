import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from scipy.stats import entropy

from datasets.smap_msl.smap_msl_dataset import SmapMslDataset
from datasets.swat.swat_data_creator import swat_sensor_names
from datasets.swat.swat_dataset import SwatDataset
from datasets.wadi.wadi_dataset import WadiDataset


def calculate_hist(aTensor, sensorName):
    plt.figure()
    plt.hist(aTensor, bins=50)
    # plt.colorbar()
    plt.savefig('resources/statistics/' + sensorName + '-histogram.png')
    # plt.show()


def calculate_entropy(aTenosr):
    result = entropy(aTenosr)
    np.savetxt('resources/statistics/entropy-mean.txt', result, delimiter=',', fmt="%f")
    plt.figure()
    plt.plot(swat_sensor_names, result, marker='.')
    plt.savefig('resources/statistics/entropy.png')
    # plt.show()


def calculate_correlation_between_two_signals(signal_1, signal_2):
    return np.corrcoef(signal_1, signal_2)


def calculate_correlation(data: Union[SwatDataset, WadiDataset, SmapMslDataset], plot_name, selected_features=None):
    data = data.data_from_sensor.numpy()

    xticklabels = yticklabels = 'auto'
    if selected_features is not None:
        data = data[:, selected_features]
        xticklabels = yticklabels = selected_features

    corrcoef_result = np.corrcoef(data, rowvar=False)
    draw_heatmap(corrcoef_result, f'corrcoef_{plot_name}', vmin=-1, vmax=1, xticklabels=xticklabels,
                 yticklabels=yticklabels)

    return corrcoef_result


def calculate_covariance(data: Union[SwatDataset, WadiDataset], plot_name, selected_features=None):
    data = data.data_from_sensor.numpy()

    xticklabels = yticklabels = 'auto'
    if selected_features is not None:
        data = data[:, selected_features]
        xticklabels = yticklabels = selected_features

    covariance_result = np.cov(data, rowvar=False)
    draw_heatmap(covariance_result, f'covariance_{plot_name}', xticklabels=xticklabels, yticklabels=yticklabels)

    return covariance_result


def calculate_covariance_and_correlation(data: Union[SwatDataset, WadiDataset], plot_name, selected_features=None):
    covariance = calculate_covariance(data, plot_name, selected_features=selected_features)
    corrcoef = calculate_correlation(data, plot_name, selected_features=selected_features)
    return covariance, corrcoef


# def draw_heatmap(data, name, vmin=None, vmax=None, xticklabels='auto', yticklabels='auto'):
#     sns.heatmap(pd.DataFrame(data), cmap='jet',
#                 square=True,
#                 linewidth=.5, cbar_kws={"shrink": .5}, vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels)
#     plt.title(f'{name}')
#     dirpath = 'resources/plots/statistics/'
#     os.makedirs(dirpath, exist_ok=True)
#     plt.savefig(f'resources/plots/statistics/{name}.png')
#     plt.close()


def calculateStatistics(aTensor, vector_length, name):
    calculate_covariance_and_correlation(aTensor, name)
    exit()
    numpyVal = aTensor.numpy()
    mean_value = np.mean(numpyVal, axis=0)
    max_value = np.max(numpyVal, axis=0)
    min_value = np.min(numpyVal, axis=0)

    plt.figure()
    plt.plot(swat_sensor_names, mean_value)
    plt.savefig('resources/statistics/mean.png')

    plt.figure()
    plt.plot(swat_sensor_names, max_value)
    plt.savefig('resources/statistics/max.png')

    plt.plot(swat_sensor_names, min_value)
    plt.savefig('resources/statistics/min.png')

    np.savetxt('resources/statistics/swat-mean.txt', mean_value, delimiter=',', fmt="%f")
    np.savetxt('resources/statistics/swat-min.txt', min_value, delimiter=',', fmt="%f")
    np.savetxt('resources/statistics/swat-max.txt', max_value, delimiter=',', fmt="%f")

    for index in range(vector_length):
        a = numpyVal[:, index]
        senor_name = swat_sensor_names[index]
        calculate_hist(a, senor_name)
    calculate_entropy(numpyVal)
