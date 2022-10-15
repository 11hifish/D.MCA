####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# Compute baseline detectors' assignment performance in F1 score.

import argparse
import numpy as np
from src.utils import load_dataset, convert_to_binary_label
from src.inne_python import pyinne
from src.cluster_metric import compute_all_clusters_score
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from isotree import IsolationForest
from gen2out.gen2out import gen2Out

from sklearn.cluster import OPTICS, KMeans
from pyclustering.cluster.xmeans import xmeans
from experiment.hyperparameters import get_hyperparameter


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--method', type=str, default='copod',
                    help='Outlier detection method')
parser.add_argument('--clustering', type=str, default='optics',
                    help='Post-processing clustering algorithm')
parser.add_argument('--dataset', type=str, default='synthetic_1_6_2',
                    help='Dataset')
parser.add_argument('--num-exp', type=int, default=5,
                    help='# random runs (only for non-deterministic methods)')
args = parser.parse_args()


X, y = load_dataset(args.dataset)
bin_y = convert_to_binary_label(y)

print(X.shape, y.shape)

psi_list, k_list = get_hyperparameter(args.method, X.shape[0])
print('psi list: ', psi_list)
print('k list: ', k_list)

num_exp = args.num_exp

# get num of clusters
unq_y, counts = np.unique(y, return_counts=True)
n_total_clusters = len(np.where(unq_y > 0)[0])

ano_cluster_y = unq_y[(unq_y > 0) & (counts > 1)]
n_micro_clusters = len(ano_cluster_y)

num_params = max(len(psi_list), len(k_list), 1)
print('num params: ', num_params)
all_f1_scores = np.zeros((num_exp, num_params, n_micro_clusters))
print('n micro clusters: {}, n total clusters (incl. single outliers): {}'.format(n_micro_clusters, n_total_clusters))

# prepare true cluster index
true_cluster_list = [np.where(y == i)[0] for i in ano_cluster_y]
num_ano = len(np.where(y > 0)[0])
print('num anos: ', num_ano)

method_name_map = {
    'copod': 'COPOD',
    'knn': 'kNN',
    'lof': 'LOF',
    'hbos': 'HBOS',
    'iforest': 'iForest',
    'sciforest': 'SCiForest',
    'iforest_star': 'iForest*',
    'sciforest_star': 'SCiForest*',
    'inne': 'iNNE',
    'gen2out': 'Gen2Out'
}


def evaluate_clustering(ano_score, exp_idx, param_idx, f1_score_rec):
    # stage 2: clustering
    # give the correct number of outliers and prepare points to be clustered
    ano_score_argsorted = np.argsort(ano_score)
    ano_idx = ano_score_argsorted[-num_ano:]
    X_ano = X[ano_idx]
    pred_cluster_list = []
    # clustering algorithm 1: optics
    if args.clustering.lower() == 'optics':
        labels_optics = OPTICS(min_samples=2).fit_predict(X_ano)
        # map to the original data
        cluster_nos = np.unique(labels_optics)
        for cno in cluster_nos:
            cno_idx = np.where(labels_optics == cno)[0]
            data_idx = ano_idx[cno_idx]
            pred_cluster_list.append(data_idx)
    # clustering algorithm 2: x means
    elif args.clustering.lower() == 'xmeans':
        model_xmeans = xmeans(X_ano)
        model_xmeans.process()
        clusters = model_xmeans.get_clusters()
        for i, cidx in enumerate(clusters):
            cidx = np.array(cidx)
            data_idx = ano_idx[cidx]
            pred_cluster_list.append(data_idx)
    # clustering algorithm 2: k means
    elif args.clustering.lower() == 'kmeans':
        model_kmeans = KMeans(n_clusters=n_total_clusters).fit(X_ano)
        labels_kmeans = model_kmeans.labels_
        cluster_nos = np.unique(labels_kmeans)
        for cno in cluster_nos:
            cno_idx = np.where(labels_kmeans == cno)[0]
            data_idx = ano_idx[cno_idx]
            pred_cluster_list.append(data_idx)
    else:
        raise Exception('Which clustering algorithm?')
    # score outlier assignment
    f1_score_all_clusters = compute_all_clusters_score(true_cluster_list, pred_cluster_list)
    f1_score_rec[exp_idx, param_idx, :] = f1_score_all_clusters


if args.method == 'gen2out':
    for exp_idx in range(num_exp):
        model = gen2Out().fit(X)
        ano_score = model.decision_function(X)
        ga_scores, ga_indices = model.group_anomaly_scores(X)
        f1_score_all_clusters = compute_all_clusters_score(true_cluster_list, ga_indices)
        all_f1_scores[exp_idx, 0, :n_micro_clusters] = f1_score_all_clusters
    print(all_f1_scores)
else:  # post-processing baseline
    for exp_idx in range(num_exp):
        if args.method.lower() == 'copod':
            # stage 1: detection
            model = COPOD().fit(X)
            ano_score = model.decision_scores_
            auc_sc = roc_auc_score(bin_y, ano_score)
            ap_sc = average_precision_score(bin_y, ano_score)
            print('data: {}, AUC: {:.4f}, AP: {:.4f}'.format(args.dataset, auc_sc, ap_sc))
            # stage 2: clustering
            evaluate_clustering(ano_score, exp_idx, 0, all_f1_scores)
        elif args.method.lower() == 'hbos':
            model = HBOS(n_bins='auto').fit(X)
            ano_score = model.decision_scores_
            auc_sc = roc_auc_score(bin_y, ano_score)
            ap_sc = average_precision_score(bin_y, ano_score)
            print('data: {}, AUC: {:.4f}, AP: {:.4f}'.format(args.dataset, auc_sc, ap_sc))
            # stage 2: clustering
            evaluate_clustering(ano_score, exp_idx, 0, all_f1_scores)
        elif args.method.lower() in ['iforest', 'inne', 'iforest_star', 'sciforest', 'sciforest_star']:
            print('psi list: ', psi_list)
            print('exp no: {}'.format(exp_idx))

            for i, psi_ in enumerate(psi_list):
                if args.method.lower() in ['iforest', 'iforest_star']:
                    print('[TREE] method: {}'.format(args.method.lower()))
                    model = IForest(n_estimators=100, max_samples=psi_).fit(X)
                    ano_score = model.decision_scores_
                elif args.method.lower() == 'inne':
                    ano_score = pyinne(X, X, t=100, psi=psi_)
                elif args.method.lower() in ['sciforest', 'sciforest_star']:
                    print('[SC-TREE] method: {}'.format(args.method.lower()))
                    model = IsolationForest(ndim=2, sample_size=int(psi_), max_depth=None,
                                            ntrees=100, missing_action="fail",
                                            coefs="normal", ntry=10,
                                            prob_pick_avg_gain=1, penalize_range=True).fit(X)
                    ano_score = model.predict(X)
                    # print('SCiForest ano score: ', ano_score)
                else:
                    raise Exception('method should not go here!')
                AUC = roc_auc_score(bin_y, ano_score)
                AP = average_precision_score(bin_y, ano_score)
                print('i: {}, psi_: {}, AUC: {:.4f}, AP: {:.4f}'.format(i, psi_, AUC, AP))

                # stage 2: clustering
                evaluate_clustering(ano_score, exp_idx, i, all_f1_scores)
        elif args.method.lower() in ['lof', 'knn']:
            print('k_list: ', k_list)
            f1_score_params = np.zeros((len(k_list), n_micro_clusters + 1))
            print('exp no: {}'.format(exp_idx))
            for i, k_ in enumerate(k_list):
                if args.method.lower() == 'lof':
                    model = LOF(n_neighbors=k_).fit(X)
                else:
                    model = KNN(n_neighbors=k_).fit(X)
                ano_score = model.decision_scores_
                AUC = roc_auc_score(bin_y, ano_score)
                AP = average_precision_score(bin_y, ano_score)
                print('i: {}, k: {}, AUC: {:.4f}, AP: {:.4f}'.format(i, k_, AUC, AP))

                # stage 2: clustering
                evaluate_clustering(ano_score, exp_idx, i, all_f1_scores)
        else:
            raise Exception('Which method??')

avg_per_cluster = np.mean(all_f1_scores, axis=(0, 1))
std_per_cluster = np.std(all_f1_scores, axis=(0, 1))
print('\n====== Results ======')
if args.method.lower() == 'gen2out':
    method_str = method_name_map[args.method.lower()] + ' '
else:
    if args.clustering.lower() == 'optics':
        clustering_name = 'OPTICS'
    else:
        clustering_name = 'X-Means'
    method_str = '{} + {} '.format(method_name_map[args.method.lower()], clustering_name)
print('Method: {}, Data: {}'.format(method_str, args.dataset))
print('Per cluster avg. F1 score (std.) across experiments: ')
score_str = ''.join(['{:.4f} ({:.4f}), '.format(avg_per_cluster[i], std_per_cluster[i]) for i in range(n_micro_clusters)])
print(score_str)
print('Avg. F1 score (std. ) across clusters and experiments: ')
print('{:.4f} ({:.4f})'.format(np.mean(all_f1_scores), np.std(all_f1_scores)))

