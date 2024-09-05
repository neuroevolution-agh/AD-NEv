import numpy as np


def median_downsample(data, ratio):
    new_nrows = int(np.floor(data.shape[0] / ratio))
    result = np.empty([new_nrows, data.shape[1]])
    for i in range(int(np.floor(data.shape[0] / ratio))):
        start = i * ratio
        end = (i + 1) * ratio
        result[i] = np.median(data[start: end], axis=0)
    return result
