####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# Compute the F1 score of outlier micro-clusters assignment.

import numpy as np


def compute_single_cluster_score(true_cluster, pred_cluster):
    if isinstance(pred_cluster, set):
        pred_cluster = np.array(list(pred_cluster))
    # x in true cluster and x in pred cluster
    tp = len(np.intersect1d(true_cluster, pred_cluster))
    # x not in true cluster, but x in pred cluster
    fp = len(np.setdiff1d(pred_cluster, true_cluster))
    # x in true cluster, but x not in pred cluster
    fn = len(np.setdiff1d(true_cluster, pred_cluster))
    # compute precision and recall
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    # compute F1 score
    f1_score = 2 * precision * recall / (precision + recall) \
        if precision + recall > 0 else 0
    return precision, recall, f1_score


def compute_all_clusters_score(true_cluster_list, pred_cluster_list):
    all_clusters_f1_score = np.zeros(len(true_cluster_list))
    for i, true_cluster in enumerate(true_cluster_list):
        max_f1_score = 0
        for pred_cluster in pred_cluster_list:
            _, _, this_f1_score = compute_single_cluster_score(true_cluster, pred_cluster)
            max_f1_score = max(max_f1_score, this_f1_score)
        # print('Cluster # {}, idx: {}, F1 score: {:.4f}'.format(i, true_cluster, max_f1_score))
        all_clusters_f1_score[i] = max_f1_score
    return all_clusters_f1_score
