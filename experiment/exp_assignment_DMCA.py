####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# Compute DMCA / DMCA_0's assignment performance in F1 score.

import argparse
import os
import numpy as np
from scipy.io import loadmat
from src.utils import load_dataset, convert_to_binary_label
from src.find_clusters import determine_threshold, find_clusters
from src.cluster_metric import compute_all_clusters_score
from experiment.hyperparameters import get_hyperparameter


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--method', type=str, default='DMCA',
                    help='Outlier detection method: {DMCA, or DMCA_0}')
parser.add_argument('--clustering', type=str, default='optics',
                    help='Post-processing clustering algorithm')
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

# get num of clusters
unq_y, counts = np.unique(y, return_counts=True)
n_total_clusters = len(np.where(unq_y > 0)[0])

ano_cluster_y = unq_y[(unq_y > 0) & (counts > 1)]
n_micro_clusters = len(ano_cluster_y)
all_f1_scores = np.zeros((args.num_exp, num_params, n_micro_clusters))
print('n micro clusters: {}, n total clusters (incl. single outliers): {}'.format(n_micro_clusters, n_total_clusters))

# prepare true cluster index
true_cluster_list = [np.where(y == i)[0] for i in ano_cluster_y]
num_ano = len(np.where(y > 0)[0])
print('num anos: ', num_ano)

for exp_idx in range(args.num_exp):
    print('psi list: ', psi_list)
    print('exp no: {}'.format(exp_idx))
    for i, psi_ in enumerate(psi_list):
        method_name = args.method
        data_path = os.path.join(args.load_path, '{}_{}_exp_{}_psi_{}.mat'
                             .format(args.dataset, method_name, exp_idx, psi_))
        D = loadmat(data_path)
        neighbors = D['neighbors']
        threshold = determine_threshold(neighbors)
        if threshold < 0:
            groups = []
        else:
            groups = find_clusters(neighbors, True, threshold)
        # score
        f1_score_all_clusters = compute_all_clusters_score(true_cluster_list, groups)
        all_f1_scores[exp_idx, i, :] = f1_score_all_clusters
        print('exp #: {}, psi: {}'.format(exp_idx, psi_))
        print('f1 score all clusters: ', f1_score_all_clusters)

avg_per_cluster = np.mean(all_f1_scores, axis=(0, 1))
std_per_cluster = np.std(all_f1_scores, axis=(0, 1))
# print(avg_per_cluster)
# print(std_per_cluster)
f1_score_mean = np.mean(all_f1_scores)
f1_score_std = np.std(all_f1_scores)

print('\n====== Results ======')
print('Method: {}, Data: {}'.format(args.method, args.dataset))
print('Per cluster avg. F1 score (std.) across experiments: ')
score_str = ''.join(['{:.4f} ({:.4f}), '.format(avg_per_cluster[i], std_per_cluster[i]) for i in range(n_micro_clusters)])
print(score_str)
print('Avg. F1 score (std. ) across clusters and experiments: ')
print('{:.4f} ({:.4f})'.format(f1_score_mean, f1_score_std))
