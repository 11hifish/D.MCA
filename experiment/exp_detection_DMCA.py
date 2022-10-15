####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# Compute DMCA / DMCA_0's detection performance in AUC / AP.

import argparse
import os
import numpy as np
from scipy.io import loadmat
from src.utils import load_dataset, convert_to_binary_label
from experiment.hyperparameters import get_hyperparameter
from sklearn.metrics import roc_auc_score, average_precision_score


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--method', type=str, default='DMCA',
                    help='Outlier detection method: {DMCA, or DMCA_0}')
parser.add_argument('--dataset', type=str, default='spiral_1_6_2',
                    help='Dataset')
parser.add_argument('--num-exp', type=int, default=5,
                    help='# random runs (only for non-deterministic methods)')
parser.add_argument('--load-path', type=str, default='results',
                    help='locations to retrieve the results')
args = parser.parse_args()

X, y = load_dataset(args.dataset)
bin_y = convert_to_binary_label(y)

psi_list, k_list = get_hyperparameter(args.method, X.shape[0])
print('psi list: ', psi_list)
num_params = max(len(psi_list), len(k_list), 1)
print('num params: ', num_params)

all_auc = np.zeros((args.num_exp, num_params))
all_ap = np.zeros((args.num_exp, num_params))

for exp_idx in range(args.num_exp):
    for i, psi_ in enumerate(psi_list):
        method_name = args.method
        data_path = os.path.join(args.load_path, '{}_{}_exp_{}_psi_{}.mat'
                             .format(args.dataset, method_name, exp_idx, psi_))
        D = loadmat(data_path)
        ano_score = D['ano_score'].ravel()
        auc = roc_auc_score(bin_y, ano_score)
        ap = average_precision_score(bin_y, ano_score)
        all_auc[exp_idx, i] = auc
        all_ap[exp_idx, i] = ap

print('all auc: ')
print(all_auc)
print('all ap: ')
print(all_ap)
print('[Model] {}, [Data] {}: AUC {:.2f} $\\pm$ {:.2f}, AP {:.2f} $\\pm$ {:.2f}'
          .format(args.method, args.dataset,
                  np.mean(all_auc), np.std(all_auc), np.mean(all_ap), np.std(all_ap)))
