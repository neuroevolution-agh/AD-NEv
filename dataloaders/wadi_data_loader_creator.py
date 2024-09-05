import pathlib

import torch
from torch.utils.data import DataLoader

from datasets.generic.overlap_windows_dataset import OverlapWindowsDataset
from datasets.wadi.wadi_data_creator import wadi_training_filenames, prepare_wadi_data, wadi_test_filenames, \
    prepare_wadi_train_test
from datasets.wadi.wadi_dataset import WadiDataset
from readers.xlsx_reader import read_data_xlsx
from statistics.StatisticGenerator import calculate_covariance_and_correlation
from statistics.pca_reduction import pca
import numpy as np


def split_wadi_validation(dataset, validation_size=0.2, part_dataset=1):
    print("start preparing wadi ds")
    end_point = int(len(dataset) * part_dataset)
    split_point = int((1 - validation_size) * end_point)
    train_data, train_labels = dataset[:split_point]
    validation_data, validation_labels = dataset[split_point:end_point]
    return WadiDataset(train_data, train_labels), WadiDataset(validation_data, validation_labels)


def split_wadi_dataset(dataset, train_size=0.2, test_size=0.2):
    print("start preparing wadi ds")
    end_point = int(len(dataset))
    split_point = int(train_size * end_point)
    train_data, train_labels = dataset[:split_point]
    validation_data, validation_labels = dataset[split_point:int(split_point + test_size * end_point)]
    return WadiDataset(train_data, train_labels), WadiDataset(validation_data, validation_labels)


def split_wadi_dataset_from_point(dataset, train_size=0.5, retrain_size=0.05, test_size=0.05):
    print("start preparing wadi ds")
    end_point = int(len(dataset))
    split_point = int(train_size * end_point)
    split_point_for_retrain = int(split_point + retrain_size * end_point)
    train_data, train_labels = dataset[split_point:split_point_for_retrain]
    validation_data, validation_labels = dataset[
                                         split_point_for_retrain:int(split_point_for_retrain + test_size * end_point)]
    return WadiDataset(train_data, train_labels), WadiDataset(validation_data, validation_labels)


def split_wadi_dataset_for_positive_and_negative_samples(dataset):
    print("start splitting for positive and negative wadi ds")
    positive_samples = []
    positive_samples_label = []
    negative_samples = []
    negative_samples_label = []
    for part in dataset:
        if part[1] == 1:
            positive_samples.append(part[0].tolist())
            positive_samples_label.append(part[1])
        else:
            negative_samples.append(part[0].tolist())
            negative_samples_label.append(part[1])

    return WadiDataset(positive_samples, positive_samples_label), WadiDataset(negative_samples, negative_samples_label)


def prepare_wadi_loaders(window_size, should_read_data_from_files):
    train_dataset, test_dataset = prepare_wadi_datasets(should_read_data_from_files)
    train_dataset = OverlapWindowsDataset(train_dataset, window_size=window_size)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=32, drop_last=True)

    test_dataset = OverlapWindowsDataset(test_dataset, window_size=window_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=True)

    return train_loader, test_loader


def prepare_wadi_datasets(_should_read_data_from_files, downsample=False, run_pca=False, only_test=False,
                          plot_statistics=False):
    training_loader_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/WADI-ds-train.pt'

    test_loader_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/WADI-ds-test.pt'

    if _should_read_data_from_files:
        training_data_from_file = read_data_xlsx(wadi_training_filenames)

        test_data_from_file = read_data_xlsx(wadi_test_filenames)

        train_input_as_matrix, label_train, test_input_as_matrix, label_test = prepare_wadi_train_test(
            training_data_from_file, test_data_from_file, downsample=downsample)

        if run_pca:
            train_input_as_matrix, test_input_as_matrix = pca(train_input_as_matrix, test_input_as_matrix)

        train_dataset = WadiDataset(train_input_as_matrix, label_train)

        test_dataset = WadiDataset(test_input_as_matrix, label_test)

        torch.save(train_dataset, training_loader_path)
        torch.save(test_dataset, test_loader_path)

    else:
        train_dataset = torch.load(training_loader_path)
        test_dataset = torch.load(test_loader_path)

    if plot_statistics:
        calculate_covariance_and_correlation(train_dataset, 'train_wadi')
        calculate_covariance_and_correlation(test_dataset, 'test_wadi')

    return train_dataset, test_dataset


if __name__ == '__main__':
    prepare_wadi_datasets(False, plot_statistics=True)
