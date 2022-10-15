####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

from scipy.io import loadmat
import os
import numpy as np


def load_dataset(dataname):
    if dataname in ['sandwich', 'spiral', 'synthetic10', 'vdensity']:
        data_loc = os.path.join('data', 'data_synth')
    elif dataname in ['letter', 'musk', 'optdigits', 'satimage-2', 'thyroid']:
        data_loc = os.path.join('data', 'data_semi_synth')
    else:
        data_loc = os.path.join('data', 'data_real')
    if '.mat' not in dataname:
        dataname = dataname + '.mat'
    data_path = os.path.join(data_loc, dataname)
    D = loadmat(data_path)
    X = D['data']
    y = D['class'].ravel()
    return X, y


def convert_to_binary_label(y):
    abn_idx = np.where(y > 0)[0]
    new_y = np.zeros(len(y))
    new_y[abn_idx] = 1
    return new_y
