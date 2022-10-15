####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# Get a unified set of hyperparameters depending on the dataset
# for each set of experiments

import numpy as np


def get_hyperparameter(method, n):
    # n: total number of data samples
    psi_list, k_list = [], []
    if method.lower() in ['iforest_star', 'sciforest_star', 'gen2out_star']:
        if n > 256:
            psi_list = np.array([256])
        else:
            psi_list = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
            psi_list = psi_list[psi_list <= 0.3 * n]
    elif method.lower() in ['knn', 'lof']:
        # k <= n
        k_list = np.array([1, 5, 10, 20, 30, 50])
        k_list = k_list[k_list < n]
    elif method.lower() in ['inne', 'iforest', 'sciforest', 'dmca_0']:
        # psi <= 0.5 n
        psi_list = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        psi_list = psi_list[psi_list <= 0.3 * n]
    elif method.lower() == 'dmca':
        psi_list = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        LB = max(32, int(0.01 * n))
        UB = min(1024, int(0.1 * n))
        print('Psi LB: {}, UB: {}'.format(LB, UB))
        psi_list = psi_list[(psi_list >= LB) & (psi_list <= UB)]
    return psi_list, k_list


def get_hyperparameter_psi_sensitivity(n):
    percentage = np.array([0.01, 0.04, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3])
    psi_list = np.array([int(p * n) for p in percentage])
    if psi_list[0] > 2:
        psi_list = np.concatenate(([2], psi_list))
        p = 2 / n
        percentage = np.concatenate(([p], percentage))
    return percentage, psi_list


def get_hyperparameter_clothes_lines(n):
    percentage = np.array([0.01, 0.05, 0.1, 0.15, 0.2])
    psi_list = np.array([int(p * n) for p in percentage])
    return percentage, psi_list
