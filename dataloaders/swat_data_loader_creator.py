import pathlib

import torch
from torch.utils.data import DataLoader

from datasets.generic.overlap_windows_dataset import OverlapWindowsDataset
from datasets.swat.swat_data_creator import prepare_swat_data, swat_test_filenames, swat_training_filenames, \
    number_of_swat_sensor, prepare_swat_train_test, prepare_swat_raw_data
from datasets.swat.swat_dataset import SwatDataset
from datasets.validation.ValidationDataSetCreator import prepare_validation_ds
from readers.xlsx_reader import read_data_xlsx
from statistics.StatisticGenerator import calculate_covariance_and_correlation
from statistics.pca_reduction import pca


def split_validation(dataset, validation_size=0.2, part_dataset=1):
    end_point = int(len(dataset)*part_dataset)
    split_point = int((1-validation_size) * end_point)
    train_data, train_labels = dataset[:split_point]
    validation_data, validation_labels = dataset[split_point:end_point]
    return SwatDataset(train_data, train_labels), SwatDataset(validation_data, validation_labels)


def prepare_swat_loaders(window_size, _should_read_data_from_files, downsample=False, run_pca=False, only_test=False):
    train_swat_ds, test_swat_ds = prepare_swat_datasets(_should_read_data_from_files, downsample=downsample,
                                                        run_pca=run_pca, only_test=only_test)

    if not only_test:
        train_dataset = OverlapWindowsDataset(train_swat_ds, window_size=window_size)
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=32, drop_last=True)
    else:
        train_loader = None

    test_dataset = OverlapWindowsDataset(test_swat_ds, window_size=window_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=True)

    return train_loader, test_loader


def prepare_swat_datasets(_should_read_data_from_files, downsample=False, run_pca=False, only_test=False, plot_statistics=False):
    training_loader_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-ds-train-downsampled.pt'

    test_loader_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-ds-test-downsampled.pt'
    if _should_read_data_from_files:
        training_data_from_file = read_data_xlsx(swat_training_filenames)

        test_data_from_file = read_data_xlsx(swat_test_filenames)

        train_input_as_matrix, label_train, test_input_as_matrix, label_test = prepare_swat_train_test(
            training_data_from_file, test_data_from_file, downsample=downsample)

        if run_pca:
            train_input_as_matrix, test_input_as_matrix = pca(train_input_as_matrix, test_input_as_matrix)

        train_swat_ds = SwatDataset(train_input_as_matrix, label_train)
        test_swat_ds = SwatDataset(test_input_as_matrix, label_test)

        torch.save(train_swat_ds, training_loader_path)
        torch.save(test_swat_ds, test_loader_path)


    else:
        if not only_test:
            train_swat_ds = torch.load(training_loader_path)
        else:
            train_swat_ds = None
        test_swat_ds = torch.load(test_loader_path)

    if plot_statistics:
        calculate_covariance_and_correlation(train_swat_ds, 'train_swat')
        calculate_covariance_and_correlation(test_swat_ds, 'test_swat')

    return train_swat_ds, test_swat_ds


def prepare_swat_loaders_with_validation(window_size, _should_read_data_from_files):
    training_loader_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-ds-train.pt'

    test_loader_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-ds-test.pt'

    validation_loader_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-ds-valid.pt'
    if _should_read_data_from_files:
        training_data_from_file = read_data_xlsx(swat_training_filenames)
        train_input_as_matrix, label_train = prepare_swat_data(training_data_from_file)

        train_swat_ds = SwatDataset(train_input_as_matrix, label_train)

        test_data_from_file = read_data_xlsx(swat_test_filenames)
        test_input_as_matrix, label_test = prepare_swat_data(test_data_from_file)
        test_swat_ds = SwatDataset(test_input_as_matrix, label_test)

        validation_data, validation_label = prepare_validation_ds(test_swat_ds, 0.1, 0.1,
                                                                  number_of_sensor=number_of_swat_sensor)

        validation_swat_ds = SwatDataset(validation_data, validation_label)

        torch.save(train_swat_ds, training_loader_path)
        torch.save(test_swat_ds, test_loader_path)
        torch.save(validation_swat_ds, validation_loader_path)

    else:
        train_swat_ds = torch.load(training_loader_path)
        test_swat_ds = torch.load(test_loader_path)
        validation_swat_ds = torch.load(validation_loader_path)

    train_dataset = OverlapWindowsDataset(train_swat_ds, window_size=window_size)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=32, drop_last=True)

    test_dataset = OverlapWindowsDataset(test_swat_ds, window_size=window_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=True)

    validation_swat_ds = OverlapWindowsDataset(validation_swat_ds, window_size=window_size)
    validation_loader = DataLoader(validation_swat_ds, shuffle=False, batch_size=1, drop_last=True)

    return train_loader, test_loader, validation_loader


def prepare_raw_swat_as_matrix(_should_read_data_from_files):
    training_loader_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-raw-matrix-train.pt'

    test_loader_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-raw-matrix-test.pt'

    training_label_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-raw-label-train.pt'

    test_label_path = pathlib.Path(
        __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-raw-label-test.pt'

    if _should_read_data_from_files:
        training_data_from_file = read_data_xlsx(swat_training_filenames)
        train_input_as_matrix, label_train = prepare_swat_raw_data(training_data_from_file)

        test_data_from_file = read_data_xlsx(swat_test_filenames)
        test_input_as_matrix, label_test = prepare_swat_raw_data(test_data_from_file)

        torch.save(train_input_as_matrix, training_loader_path)
        torch.save(label_train, training_label_path)
        torch.save(test_input_as_matrix, test_loader_path)
        torch.save(label_test, test_label_path)

    else:
        train_input_as_matrix = torch.load(training_loader_path)
        test_input_as_matrix = torch.load(test_loader_path)
        label_train = torch.load(training_label_path)
        label_test = torch.load(test_label_path)

    return train_input_as_matrix, label_train, test_input_as_matrix, label_test


if __name__ == '__main__':
    train_ds, test_ds = prepare_swat_datasets(False, plot_statistics=True)
    calculate_covariance_and_correlation(train_ds, plot_name='train_swat_group_0', selected_features=list(range(4, 8)))
