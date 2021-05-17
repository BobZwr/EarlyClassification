import torch

import numpy as np

def shuffle(data, label):
    assert len(data) == len(label), 'The length of data and label must be same!'

    index = np.random.permutation(len(label))
    data = data[index]
    label = label[index]

    return data, label