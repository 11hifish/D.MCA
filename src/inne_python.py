####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# This script has the Python version of iNNE (isolation using Nearest Neighbour Ensemble).
# Adapted from the Matlab version of iNNE released by one of iNNE's authors at: https://github.com/zhuye88/iNNE

import numpy as np
from scipy.spatial.distance import cdist


def pyinne_single(X_train, X_test, psi, normalize=False):
    """
        A single iNNE model.
        X_train: training set (subsample)
        X_test: testing set
        psi: subsample size (>= 2)
        normalize: return normalized iscore
    """
    if psi <= 1:
        raise Exception('subsample size should be >= 2!')
    center_id = np.random.choice(np.arange(X_train.shape[0]),
                                 size=psi, replace=False)
    centers = X_train[center_id]

    # get center's 1 NN
    center_dists = cdist(centers, centers, metric='euclidean')
    # mask diagonal (self-distance does not count)
    np.fill_diagonal(center_dists, np.inf)
    center_1nn_idx = np.argmin(center_dists, axis=1)
    # center's 1nn dist (can be 0 if there are repetitive centers)
    center_1nn_dist = center_dists[np.arange(centers.shape[0]), center_1nn_idx]
    # center's 1nn id
    center_1nn_id = center_id[center_1nn_idx]
    # center's 1nn's 1nn's dist
    center_1nn_1nn_dist = center_dists[center_1nn_idx,
                                       center_1nn_idx[center_1nn_idx]]
    # score testing samples
    all_dists = cdist(X_test, centers, metric='euclidean')
    # find testing sample's corresponding hypersphere
    mask = all_dists > center_1nn_dist  # outside hypersphere
    test_nn_center_idx = np.array([(lambda x, m: -1 if m.all() else np.argmin(x))
                                   (np.ma.masked_array(all_dists[i], mask=mask[i]), mask[i])
                                   for i in range(X_test.shape[0])])  # {-1} union [0, psi)
    if normalize:
        min_rad = np.min(center_1nn_dist)
        print('all rad: {}, 1nn 1nn dist: {}, min rad: {}'.format(center_1nn_dist, center_1nn_1nn_dist, min_rad))
        iscore = np.array([(
                               lambda j: 1. if j < 0 else
                               (0. if center_1nn_dist[j] < 1e-7 or center_1nn_dist[j] == min_rad
                                else 1. - (center_1nn_1nn_dist[j] - min_rad) / (center_1nn_dist[j] - min_rad))
                           )
                           (test_nn_center_idx[i]) for i in range(X_test.shape[0])])
    else:
        iscore = np.array([(
                               lambda j: 1. if j < 0 else
                                (0. if center_1nn_dist[j] < 1e-7 else 1. - center_1nn_1nn_dist[j] / center_1nn_dist[j])
                            )
                           (test_nn_center_idx[i]) for i in range(X_test.shape[0])])
    return center_id, center_1nn_id, center_1nn_dist, iscore, all_dists


def pyinne(X_train, X_test, t, psi):
    all_ano_scores = np.zeros((t, X_test.shape[0]))
    for model_idx in range(t):
        _, _, _, ano_score, _ = pyinne_single(X_train=X_train, X_test=X_test, psi=psi, normalize=False)
        all_ano_scores[model_idx] = ano_score
    avg_ano_score = np.mean(all_ano_scores, axis=0)
    return avg_ano_score
