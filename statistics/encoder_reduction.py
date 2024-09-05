import pathlib

import torch
from torch import tensor

from dataloaders.swat_data_loader_creator import prepare_swat_loaders
from datasets.swat.swat_dataset import SwatDataset

training_loader_path = pathlib.Path(
    __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-ds-train-reduced-via-encoder.pt'

test_loader_path = pathlib.Path(
    __file__).parent.absolute().parent.absolute() / 'resources/dataloader/SWAT-ds-test-reduced-via-encoder.pt'


def reduce_with_encoder(encoder, train_loader, test_loader, flatten):
    cpu_device = torch.device('cpu')
    gpu_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder.to(gpu_device)

    all_data, all_labels = _reduce_for_loader(train_loader, encoder, gpu_device, cpu_device, flatten)
    reduced_train_ds = SwatDataset(all_data, all_labels)
    torch.save(reduced_train_ds, training_loader_path)
    print('Training processed')

    all_data, all_labels = _reduce_for_loader(test_loader, encoder, gpu_device, cpu_device, flatten)
    reduced_test_ds = SwatDataset(all_data, all_labels)
    torch.save(reduced_test_ds, test_loader_path)
    print('Testing processed')


def _reduce_for_loader(loader, encoder, gpu_device, cpu_device, flatten):
    all_data = []
    all_labels = []

    for data, labels in loader:
        if flatten:
            data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        data = data.to(gpu_device)
        encoding = encoder(data).detach()
        encoding = encoding.to(cpu_device)
        all_data.extend(encoding)
        all_labels.extend(labels)
    all_data_cat = torch.cat(all_data)
    return all_data_cat, all_labels


if __name__ == '__main__':
    window_size = 12
    train_loader, test_loader = prepare_swat_loaders(window_size, False)
    encoder = torch.load(f'resources/models/usad-SWAT-encoder.pt')

    reduce_with_encoder(encoder, train_loader, test_loader, flatten=True)

