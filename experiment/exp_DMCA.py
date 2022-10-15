####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# Experiments running DMCA / DMCA_0.

import argparse
import os
import numpy as np
from src.utils import load_dataset, convert_to_binary_label
from src.DMCA import DMCA_0, DMCA
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.io import savemat
from src.find_clusters import determine_threshold, find_clusters
from experiment.hyperparameters import get_hyperparameter


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--method', type=str, default='DMCA',
                    help='Outlier detection method: {DMCA, or DMCA_0}')
parser.add_argument('--dataset', type=str, default='synthetic10',
                    help='Dataset')
parser.add_argument('--num-exp', type=int, default=5,
                    help='# random runs (only for non-deterministic methods)')
parser.add_argument('--save-path', type=str, default=None,
                    help='locations to save the results')
args = parser.parse_args()


if args.save_path is not None and not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)

# prepare dataset
X, y = load_dataset(args.dataset)
bin_y = convert_to_binary_label(y)

# hyperparameters
psi_list, k_list = get_hyperparameter(args.method, X.shape[0])
t = 100
print('psi list: ', psi_list)
print('k list: ', k_list)


# start training
all_auc = np.zeros(args.num_exp)
all_ap = np.zeros(args.num_exp)

for exp_idx in range(args.num_exp):
    param_auc = np.zeros(len(psi_list))
    param_ap = np.zeros(len(psi_list))
    for i, psi_ in enumerate(psi_list):
        if args.method == 'DMCA':
            ano_score, neighbor_recorder = DMCA(X=X, t=t, psi=psi_)
        elif args.method == 'DMCA_0':
            ano_score, neighbor_recorder = DMCA_0(X=X, t=t, psi=psi_)
        else:
            raise Exception('Which method ???')
        # save the results
        if args.save_path is not None:
            save_dic = {
                'ano_score': ano_score,
                'neighbors': neighbor_recorder
            }
            savemat(os.path.join(args.save_path, '{}_{}_exp_{}_psi_{}.mat'
                                 .format(args.dataset, args.method, exp_idx, psi_)), save_dic)
        param_auc[i] = roc_auc_score(bin_y, ano_score)
        param_ap[i] = average_precision_score(bin_y, ano_score)
        print('i: {}, psi: {}, AUC: {:.4f}, AP: {:.4f}'.format(i, psi_, param_auc[i], param_ap[i]))
        threshold = determine_threshold(neighbor_recorder)
        groups = find_clusters(neighbor_recorder, True, threshold)
        print('groups: ', groups)
    all_auc[exp_idx] = np.mean(param_auc)
    all_ap[exp_idx] = np.mean(param_ap)

print('[Model] {}, [Data] {}: AUC {:.4f} ({:.4f}), AP {:.4f} ({:.4f})'
      .format(args.method, args.dataset,
              np.mean(all_auc), np.std(all_auc), np.mean(all_ap), np.std(all_ap)))

