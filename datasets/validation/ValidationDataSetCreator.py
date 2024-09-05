import numpy as np


def prepare_validation_ds(test_dataset, percentage_of_data, percentage_of_anomalies, number_of_sensor):
    len_of_validation_ds = int(percentage_of_data * len(test_dataset.labels))
    len_of_anomalies = int(percentage_of_anomalies * len_of_validation_ds)

    validation_data = np.empty([len_of_validation_ds, number_of_sensor])
    validation_labels = np.empty(len_of_validation_ds)

    indexes_of_anomalies = np.where(test_dataset.labels == 1)
    indexes_of_normal = np.where(test_dataset.labels == 0)

    for i in range(len_of_validation_ds - len_of_anomalies):
        index_of_normal = indexes_of_normal[0][i]
        validation_data[i], validation_labels[i] = test_dataset.__getitem__(index_of_normal)

    for i in range(len_of_anomalies):
        index_of_anomaly = indexes_of_anomalies[0][i]
        validation_data[i + (len_of_validation_ds - len_of_anomalies)], validation_labels[
            i + (len_of_validation_ds - len_of_anomalies)] \
            = test_dataset.__getitem__(index_of_anomaly)

    return validation_data, validation_labels
