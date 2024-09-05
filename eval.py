import argparse
from typing import Tuple, List

import torch
from sklearn.metrics import classification_report

from CONSTANT import SWAT_GROUPS, WADI_GROUPS, WADI_WINDOW_SIZE, SWAT_WINDOW_SIZE, WADI_THRESHOLD, SWAT_THRESHOLD
from dataloaders.selected_features_loader import prepare_loaders_with_selected_features
from dataloaders.swat_data_loader_creator import prepare_swat_datasets
from dataloaders.wadi_data_loader_creator import prepare_wadi_datasets

_should_read_data_from_files = True


def predict(model, loss_function, test_loader, device) -> Tuple[List, List]:
    model.eval()
    y_true = []
    errors = []

    model.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs = model(data)
            loss = loss_function(outputs, data)
            errors.append(loss)
            y_true.append(target)
            data.detach().cpu()
            del data
            del target

    y_true_transformed = [tensor.numpy()[0] for tensor in y_true]
    return y_true_transformed, errors


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default='swat',
        help="dataset name, could be swat or wadi",
    )

    args = parser.parse_args()
    dataset = args.dataset

    final_model_name = f'resources/models/{dataset}/model.pt'
    window_size = SWAT_WINDOW_SIZE if dataset == 'swat' else WADI_WINDOW_SIZE
    threshold = SWAT_THRESHOLD if dataset == 'swat' else WADI_THRESHOLD
    groups = SWAT_GROUPS if dataset == 'swat' else WADI_GROUPS

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_function = torch.nn.MSELoss()

    if dataset == 'swat':
        train_ds, test_ds = prepare_swat_datasets(_should_read_data_from_files=False, downsample=True)
    elif dataset == 'wadi':
        train_ds, test_ds = prepare_wadi_datasets(_should_read_data_from_files=False, downsample=True)
    else:
        raise NotImplemented('method is not implemented yet ', dataset)

    train_loader, test_loader = prepare_loaders_with_selected_features(train_ds, test_ds, window_size=window_size,
                                                                       selected_features=groups)
    model = torch.load(final_model_name)
    y_true, errors = predict(model, loss_function, test_loader, device=device)
    y_predicted = [1 if e >= threshold else 0 for e in errors]
    report = classification_report(y_true, y_predicted, output_dict=True)
    current_f1 = report['1']['f1-score']
    print(f'Current best f1 is {current_f1}')
