####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# This script is based on DMCA.py,
# which computes the masking effect across iterations/models
# in addition to outlier detection and micro-clusters assignment

import numpy as np
from src.inne_python import pyinne_single
from src.DMCA import split_two_phases, DMCA_single_model_subroutine
from src.find_clusters import determine_threshold, find_clusters
from scipy.sparse import csr_matrix

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


def count_masking_effect(center_id, ass):
    # define 1 masking effect = 1 cluster get >= 2 subsamples
    center_ass = ass[center_id]
    ano_center_idx = np.where(center_ass > 0)[0]
    if len(ano_center_idx) == 0:  # only normal centers
        return 0
    else:
        ano_center_ass = center_ass[ano_center_idx]
        uc, ct = np.unique(ano_center_ass, return_counts=True)
        return len(np.where(ct >= 2)[0])


def inne_masking_effect(X, t, psi, y):
    # y: multiclass label
    all_masking_effect = np.zeros(t)
    for model_idx in range(t):
        center_id, center_1nn_id, center_1nn_dist, iscore, all_dists = pyinne_single(X_train=X, X_test=X, psi=psi, normalize=False)
        me_num = count_masking_effect(center_id=center_id, ass=y)
        all_masking_effect[model_idx] = me_num
    return all_masking_effect


def DMCA_masking_effect(X, t, psi, y, p=0.1):
    n = X.shape[0]
    num_prune = int(p * n)
    # split t into 2 phases and determine the psi values for phase 1
    phase1_t, phase1_psis = split_two_phases(t, psi)
    neighbor_recorder = csr_matrix((n, n), dtype=np.int32)
    all_masking_effect = np.zeros(t)
    ## Phase 1: vary psi, only to get cluster info, no pruning
    for model_idx in range(phase1_t):
        current_psi = phase1_psis[model_idx]
        print('[Model {}], current psi: {}'.format(model_idx, current_psi))
        center_id, center_1nn_id, center_1nn_dist, iscore, all_dists = pyinne_single(X, X, current_psi)
        all_masking_effect[model_idx] = count_masking_effect(center_id, y)
        _, neighbor_recorder = DMCA_single_model_subroutine(X=X, center_id=center_id, ano_scores=iscore,
                                                            num_prune=num_prune, psi=current_psi,
                                                            neighbor_recorder=neighbor_recorder)
    # end of phase 1, get clusters/groups and ready to prune in phase 2
    threshold = determine_threshold(neighbor_recorder)
    if threshold < 0:
        groups = {}
    else:
        groups = find_clusters(neighbor_recorder=neighbor_recorder, only_cluster=True, threshold=threshold)
    print('groups: ', groups)
    if len(groups) == 0:
        R_idx = np.arange(n)
    else:
        prune_pt_id = np.array(list(set.union(*groups)))
        R_idx = np.setdiff1d(np.arange(n), prune_pt_id)
    ## Phase 2: fix psi, start pruning
    iscore_phase2 = np.zeros((n, t - phase1_t))
    for model_idx in range(phase1_t, t):
        print('[Model {}], phase 2 psi: {}'.format(model_idx, psi))
        R = X[R_idx]
        center_id, center_1nn_id, center_1nn_dist, iscore, all_dists = pyinne_single(R, X, psi)
        all_masking_effect[model_idx] = count_masking_effect(center_id, y)
        iscore_phase2[:, model_idx - phase1_t] = iscore
        # get mean iscore upto date and compute F
        iscore_upto_current = np.mean(iscore_phase2[:, :model_idx + 1], axis=1)
        R_idx, neighbor_recorder = DMCA_single_model_subroutine(X=X, center_id=center_id,
                                                                ano_scores=iscore_upto_current,
                                                                num_prune=num_prune, psi=psi,
                                                                neighbor_recorder=neighbor_recorder)
    # end of phase 2
    return all_masking_effect


def DMCA_0_masking_effect(X, t, psi, y, p=0.1):
    """
        Neighor anomalous scores + change point detection
        X: training and testing data
        t: # ensembles
        psi: subsample size
        p: percentage of data with the highest anomalous score to be considered in F
    """
    n = X.shape[0]
    num_prune = int(p * n)
    iscore_all = np.zeros((n, t))
    R_idx = np.arange(n)
    neighbor_recorder = csr_matrix((n, n), dtype=np.int32)
    all_masking_effect = np.zeros(t)

    for model_idx in range(t):
        print('[Model {}]'.format(model_idx))
        R = X[R_idx]
        center_id, center_1nn_id, center_1nn_dist, iscore, all_dists = pyinne_single(R, X, psi)
        all_masking_effect[model_idx] = count_masking_effect(center_id, y)
        iscore_all[:, model_idx] = iscore
        # get mean iscore upto date
        iscore_upto_current = np.mean(iscore_all[:, :model_idx + 1], axis=1)
        # decide on TPs and mark neighbors
        R_idx, neighbor_recorder = DMCA_single_model_subroutine(X=X, center_id=center_id,
                                                                ano_scores=iscore_upto_current,
                                                                num_prune=num_prune, psi=psi,
                                                                neighbor_recorder=neighbor_recorder)
    # find groups in the end
    return all_masking_effect

